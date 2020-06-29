# (C) Datadog, Inc. 2020-present
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)
import json
from collections import defaultdict

import mmh3
import psycopg2
import psycopg2.extras

from datadog_checks.base import AgentCheck, ConfigurationError, is_affirmative

try:
    import datadog_agent
except ImportError:
    from ..stubs import datadog_agent

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

db_postgres_event_keys = [
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


# TODO: move this to a shared lib
def compute_sql_signature(normalized_query):
    """
    Generate a 64-bit hex signature on the obfuscated & normalized query.
    """
    return format(mmh3.hash64(normalized_query, signed=False)[0], 'x')


def compute_exec_plan_signature(normalized_json_plan):
    """
    Given a normalized json query execution plan, generate its 64-bit hex signature
    """
    if not normalized_json_plan:
        return None
    with_sorted_keys = json.dumps(json.loads(normalized_json_plan), sort_keys=True)
    return format(mmh3.hash64(with_sorted_keys, signed=False)[0], 'x')


class PgStatementsMixin(object):
    """
    Mixin for collecting telemetry on executed statements.
    """

    def __init__(self, *args, **kwargs):
        # Cache results of monotonic pg_stat_statements to compare to previous collection
        self._statements_cache = {}

        # Available columns will be queried once and cached as the source of truth.
        self.__pg_stat_statements_columns = None

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

    def _collect_activity_by_datname(self, total_duration_seconds=5):
        """

        :param total_duration_seconds:
        :return: {(sql_signature) -> row}
        """
        query = """
        SELECT * FROM pg_stat_activity
        WHERE datname IS NOT NULL
        AND coalesce(TRIM(query), '') != '';
        """

        # TODO: redo this several times
        activity_by_datname = defaultdict(list)

        with self.db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            rows = self._execute_query(cursor, query)
            if not rows:
                return activity_by_datname

        max_query_start = None
        for row in rows:
            if not row['query'] or not row['datname']:
                continue
            activity_by_datname[row['datname']].append(row)
            if max_query_start is None or row['query_start'] > max_query_start:
                max_query_start = row['query_start']

        return activity_by_datname

    def _run_explain(self, db, statement):
        # TODO: cleaner query cleaning to strip comments, etc.
        if statement.strip().split(' ', 1)[0].lower() not in VALID_EXPLAIN_STATEMENTS:
            return None

        with db.cursor() as cursor:
            try:
                query = 'EXPLAIN (FORMAT JSON) {statement}'.format(statement=statement)
                cursor.execute(query)
                result = cursor.fetchone()
            except Exception as e:
                self.log.error("failed to collect execution plan for query='%s': %s", statement, e)
                return None

        if not result or len(result) < 1 or len(result[0]) < 1:
            return None

        # TODO: see if there's a way we can avoid having the client library pre-emptively deserialize the json
        # plan for us
        plan = result[0][0]
        return json.dumps(plan) if isinstance(plan, dict) else plan

    def _collect_execution_plans(self, instance_tags):
        seen_statements = set()
        seen_statement_plan_sigs = set()
        events = []
        for datname, samples in self._collect_activity_by_datname().items():
            try:
                db = self._lazy_connect_database(datname)
            except Exception as e:
                self.log.warn("skipping execution plan collection for database=%s due to failed connection: %s",
                              datname, e)
                continue

            for row in samples:
                original_statement = row['query']
                if original_statement in seen_statements:
                    continue
                seen_statements.add(original_statement)
                obfuscated_statement = datadog_agent.obfuscate_sql(original_statement)
                query_signature = compute_sql_signature(obfuscated_statement)
                plan = self._run_explain(db, original_statement)
                if not plan:
                    continue
                normalized_plan = datadog_agent.obfuscate_sql_exec_plan(plan, normalize=True) if plan else None
                obfuscated_plan = datadog_agent.obfuscate_sql_exec_plan(plan) if plan else None
                plan_signature = compute_exec_plan_signature(normalized_plan)
                statement_plan_sig = (query_signature, plan_signature)
                if statement_plan_sig not in seen_statement_plan_sigs:
                    seen_statement_plan_sigs.add(statement_plan_sig)
                    events.append({
                        'db': {
                            # 'instance': None,
                            'statement': obfuscated_statement,
                            'query_signature': query_signature,
                            'plan': plan,
                            # 'plan_cost': None,
                            'plan_signature': plan_signature,
                            'debug': {
                                'normalized_plan': normalized_plan,
                                'obfuscated_plan': obfuscated_plan,
                                'original_statement': original_statement,
                            },
                            'postgres': {k: row[k] for k in db_postgres_event_keys if k in row},
                        }
                    })
                    self.log.info("event: %s", events[-1])
