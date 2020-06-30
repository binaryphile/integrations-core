# (C) Datadog, Inc. 2020-present
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)
import decimal
import json
from collections import defaultdict

import mmh3
import psycopg2
import psycopg2.extras
import time
from datadog_checks.base import AgentCheck
import socket
from datadog_checks.base.utils.db.sql import compute_sql_signature, compute_exec_plan_signature

from contextlib import closing

try:
    import datadog_agent
except ImportError:
    from ..stubs import datadog_agent


class EventEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(EventEncoder, self).default(o)


# TODO: As a future optimization, the `query` column should be cached on first request
# and not requested again until agent restart or pg_stats reset. This is to avoid
# hitting disk every poll period. This is fine when the query is run every 15s,
# but not when it is run every 100ms for instance. Full queries can be quite large.
STATEMENTS_QUERY = """
SELECT {cols}
  FROM pg_stat_statements
  LEFT JOIN pg_roles
         ON pg_stat_statements.userid = pg_roles.oid
  LEFT JOIN pg_database
         ON pg_stat_statements.dbid = pg_database.oid
ORDER BY (pg_stat_statements.total_time / NULLIF(pg_stat_statements.calls, 0)) DESC;
"""

# Required columns for the check to run
PG_STAT_STATEMENTS_REQUIRED_COLUMNS = frozenset({
    'calls',
    'query',
    'total_time',
    'rows',
})

# Count columns to be converted to metrics
PG_STAT_STATEMENTS_METRIC_COLUMNS = {
    'calls': ('pg_stat_statements.calls', 'postgresql.queries.count', AgentCheck.count),
    'total_time': ('pg_stat_statements.total_time', 'postgresql.queries.time', AgentCheck.count),
    'rows': ('pg_stat_statements.rows', 'postgresql.queries.rows', AgentCheck.count),
    'shared_blks_hit': ('pg_stat_statements.shared_blks_hit', 'postgresql.queries.shared_blks_hit', AgentCheck.count),
    'shared_blks_read': (
        'pg_stat_statements.shared_blks_read', 'postgresql.queries.shared_blks_read', AgentCheck.count),
    'shared_blks_dirtied': (
        'pg_stat_statements.shared_blks_dirtied', 'postgresql.queries.shared_blks_dirtied', AgentCheck.count),
    'shared_blks_written': (
        'pg_stat_statements.shared_blks_written', 'postgresql.queries.shared_blks_written', AgentCheck.count),
    'local_blks_hit': ('pg_stat_statements.local_blks_hit', 'postgresql.queries.local_blks_hit', AgentCheck.count),
    'local_blks_read': ('pg_stat_statements.local_blks_read', 'postgresql.queries.local_blks_read', AgentCheck.count),
    'local_blks_dirtied': (
        'pg_stat_statements.local_blks_dirtied', 'postgresql.queries.local_blks_dirtied', AgentCheck.count),
    'local_blks_written': (
        'pg_stat_statements.local_blks_written', 'postgresql.queries.local_blks_written', AgentCheck.count),
    'temp_blks_read': ('pg_stat_statements.temp_blks_read', 'postgresql.queries.temp_blks_read', AgentCheck.count),
    'temp_blks_written': (
        'pg_stat_statements.temp_blks_written', 'postgresql.queries.temp_blks_written', AgentCheck.count),
}

# Columns to apply as tags
PG_STAT_STATEMENTS_TAG_COLUMNS = {
    'datname': ('pg_database.datname', 'db'),
    'rolname': ('pg_roles.rolname', 'user'),
    'query': ('pg_stat_statements.query', 'query'),
}

# Transformation functions to apply to each column before submission
PG_STAT_STATEMENTS_TRANSFORM = {
    # obfuscate then truncate the query to 200 chars
    'query': lambda q: datadog_agent.obfuscate_sql(q)[:200],
}

VALID_EXPLAIN_STATEMENTS = frozenset({
    'select',
    'table',
    'delete',
    'insert',
    'replace',
    'update',
})

# keys from pg_stat_activity to include along with each (sample & execution plan)
pg_stat_activity_sample_keys = [
    'query_start',
    'datname',
    'usesysid',
    'application_name',
    'client_addr',
    'client_port',
    'wait_event_type',
    'wait_event',
    'state',
]


class PgStatementsMixin(object):
    """
    Mixin for collecting telemetry on executed statements.
    """

    def __init__(self, *args, **kwargs):
        # Cache results of monotonic pg_stat_statements to compare to previous collection
        self._statements_cache = {}

        # Available columns will be queried once and cached as the source of truth.
        self.__pg_stat_statements_columns = None
        self._activity_last_query_start = None

    def _execute_query(self, cursor, query, log_func=None):
        raise NotImplementedError('Check must implement _execute_query()')

    def _lazy_connect_database(self, dbname):
        raise NotImplementedError('Check must implement _lazy_connect_database()')

    @property
    def _pg_stat_statements_columns(self):
        """
        Lazy-loaded list of the columns available under the `pg_stat_statements` table. This must be done because
        version is not a reliable way to determine the available columns on `pg_stat_statements`. The database
        can be upgraded without upgrading extensions, even when the extension is included by default.
        """
        if self.__pg_stat_statements_columns is not None:
            return self.__pg_stat_statements_columns
        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'pg_stat_statements';
            """
        cursor = self.db.cursor()
        columns = self._execute_query(cursor, query)
        self.__pg_stat_statements_columns = frozenset(column[0] for column in columns)
        return self.__pg_stat_statements_columns

    def _collect_statement_metrics(self, instance_tags):
        # Sanity checks
        missing_columns = PG_STAT_STATEMENTS_REQUIRED_COLUMNS - self._pg_stat_statements_columns
        if len(missing_columns) > 0:
            self.log.warning('Unable to collect statement metrics because required fields are unavailable: {}'.format(
                missing_columns))
            return

        cursor = self.db.cursor(cursor_factory=psycopg2.extras.DictCursor)
        columns = []
        for entry in (PG_STAT_STATEMENTS_METRIC_COLUMNS, PG_STAT_STATEMENTS_TAG_COLUMNS):
            for alias, (column, *_) in entry.items():
                # Only include columns which are available on the table
                if alias in self._pg_stat_statements_columns or alias in PG_STAT_STATEMENTS_TAG_COLUMNS:
                    columns.append('{column} AS {alias}'.format(column=column, alias=alias))

        rows = self._execute_query(cursor, STATEMENTS_QUERY.format(cols=', '.join(columns)))
        if not rows:
            return
        rows = rows[:self.config.max_query_metrics]

        new_cache = {}
        for row in rows:
            key = (row['rolname'], row['datname'], row['query'])
            new_cache[key] = row
            if key not in self.statement_cache:
                continue
            prev = self.statement_cache[key]

            # pg_stats reset will cause this
            if row['calls'] - prev['calls'] <= 0:
                continue

            obfuscated_query = datadog_agent.obfuscate_sql(row['query'])
            tags = ['query_signature:' + compute_sql_signature(obfuscated_query)] + instance_tags
            for tag_column, (_, alias) in PG_STAT_STATEMENTS_TAG_COLUMNS.items():
                if tag_column not in self._pg_stat_statements_columns:
                    continue
                value = PG_STAT_STATEMENTS_TRANSFORM.get(tag_column, lambda x: x)(row[tag_column])
                tags.append('{alias}:{value}'.format(alias=alias, value=value))

            for alias, (_, name, fn) in PG_STAT_STATEMENTS_METRIC_COLUMNS.items():
                if alias not in self._pg_stat_statements_columns:
                    continue

                val = row[alias] - prev[alias]

                fn(self, name, val, tags=tags)

        self.statement_cache = new_cache

    def _sample_pg_stat_activity_by_database(self, instance_tags=None):
        start_time = time.time()
        query = """
        SELECT * FROM pg_stat_activity
        WHERE datname IS NOT NULL
        AND coalesce(TRIM(query), '') != ''
        """

        total_rows = 0
        activity_by_database = defaultdict(list)
        for _ in range(self.config.pg_stat_activity_samples_per_run):
            if time.time() - start_time > self.config.pg_stat_activity_plan_collect_time_limit:
                self.log.debug("exceeded pg_stat_activity plan collection time limit of %s s",
                               self.config.pg_stat_activity_plan_collect_time_limit)
                break
            if total_rows > self.config.pg_stat_activity_sampled_row_limit:
                self.log.debug("exceeded pg_stat_activity total row limit of %s row",
                               self.config.pg_stat_activity_sampled_row_limit)
                break
            sample_start_time = time.time()
            self.db.rollback()
            with self.db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                if self._activity_last_query_start:
                    cursor.execute(query + " AND query_start > %s", (self._activity_last_query_start,))
                else:
                    cursor.execute(query)
                rows = cursor.fetchall()
            # TODO: once stable, either remove these development metrics or make them configurable in a debug mode
            self.histogram("dd.postgres.sample_pg_stat_activity.sample.time", (time.time() - sample_start_time) * 1000,
                           tags=instance_tags)
            self.histogram("dd.postgres.sample_pg_stat_activity.sample.rows", len(rows), tags=instance_tags)
            for row in rows:
                if not row['query'] or not row['datname']:
                    continue
                activity_by_database[row['datname']].append(row)
                if self._activity_last_query_start is None or row['query_start'] > self._activity_last_query_start:
                    self._activity_last_query_start = row['query_start']
                total_rows += 1
            time.sleep(self.config.pg_stat_activity_sleep_per_sample)

        self.gauge("dd.postgres.sample_pg_stat_activity.total.time", (time.time() - start_time) * 1000,
                   tags=instance_tags)
        return activity_by_database

    def _run_explain(self, db, statement, instance_tags=None):
        # TODO: cleaner query cleaning to strip comments, etc.
        if statement.strip().split(' ', 1)[0].lower() not in VALID_EXPLAIN_STATEMENTS:
            return None
        with db.cursor() as cursor:
            try:
                start_time = time.time()
                query = 'EXPLAIN (FORMAT JSON) {statement}'.format(statement=statement)
                cursor.execute(query)
                result = cursor.fetchone()
                self.histogram("dd.postgres.run_explain.time", (time.time() - start_time) * 1000, tags=instance_tags)
            except Exception as e:
                self.log.error("failed to collect execution plan for query='%s': %s", statement, e)
                return None
        if not result or len(result) < 1 or len(result[0]) < 1:
            return None
        return result[0][0]

    def _submit_log_events(self, events):
        # TODO: This is a temporary hack to send logs via the Python integrations and requires customers
        # to configure a TCP log on port 10518. THIS CODE SHOULD NOT BE MERGED TO MASTER
        try:
            with closing(socket.create_connection(('localhost', 10518))) as c:
                for e in events:
                    c.sendall((json.dumps(e, cls=EventEncoder, default=str) + '\n').encode())
        except ConnectionRefusedError:
            self.warning('Unable to connect to the logs agent; please see the '
                         'documentation on configuring the logs agent.')
            return

    def _collect_execution_plans(self, instance_tags):
        start_time = time.time()
        # avoid reprocessing the exact same statement
        seen_statements = set()
        # keep only one sample per unique (query, plan)
        seen_statement_plan_sigs = set()
        events = []
        samples_by_database = self._sample_pg_stat_activity_by_database(instance_tags=instance_tags)
        plan_collection_start_time = time.time()
        for database, samples in samples_by_database.items():
            try:
                db = self._lazy_connect_database(database)
            except Exception as e:
                self.log.warn("skipping execution plan collection for database=%s due to failed connection: %s",
                              database, e)
                continue

            for row in samples:
                original_statement = row['query']
                if original_statement in seen_statements:
                    continue
                seen_statements.add(original_statement)
                obfuscated_statement = datadog_agent.obfuscate_sql(original_statement)
                query_signature = compute_sql_signature(obfuscated_statement)
                plan_dict = self._run_explain(db, original_statement, instance_tags)
                if not plan_dict:
                    continue
                plan = json.dumps(plan_dict)
                normalized_plan = datadog_agent.obfuscate_sql_exec_plan(plan, normalize=True) if plan else None
                obfuscated_plan = datadog_agent.obfuscate_sql_exec_plan(plan) if plan else None
                plan_signature = compute_exec_plan_signature(normalized_plan)
                statement_plan_sig = (query_signature, plan_signature)
                if statement_plan_sig not in seen_statement_plan_sigs:
                    seen_statement_plan_sigs.add(statement_plan_sig)
                    events.append({
                        'db': {
                            'instance': row['datname'],
                            'statement': obfuscated_statement,
                            'query_signature': query_signature,
                            'plan': plan,
                            'plan_cost': (plan_dict.get('Plan', {}).get('Total Cost', 0.) or 0.),
                            'plan_signature': plan_signature,
                            'debug': {
                                'normalized_plan': normalized_plan,
                                'obfuscated_plan': obfuscated_plan,
                                'original_statement': original_statement,
                            },
                            'postgres': {k: row[k] for k in pg_stat_activity_sample_keys if k in row},
                        }
                    })
        self.gauge("dd.postgres.collect_execution_plans.plans_only.time",
                   (time.time() - plan_collection_start_time) * 1000, tags=instance_tags)

        self._submit_log_events(events)
        self.gauge("dd.postgres.collect_execution_plans.total.time", (time.time() - start_time) * 1000,
                   tags=instance_tags)
        self.gauge("dd.postgres.collect_execution_plans.seen_statements", len(seen_statements), tags=instance_tags)
        self.gauge("dd.postgres.collect_execution_plans.seen_statement_plan_sigs", len(seen_statement_plan_sigs),
                   tags=instance_tags)
