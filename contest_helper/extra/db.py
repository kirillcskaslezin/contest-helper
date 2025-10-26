from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import csv
import random as rd

# Reuse value generators from the core module
# Adjust the import path if this file lives in a different package structure
from ..values import Value, Lambda, RandomValue


# ---- Column specification ----------------------------------------------------

@dataclass
class ColumnSpec:
    """Column configuration for a database table.

    Parameters:
        generator: A Value-like object (callable without args) that produces values
                   for the column. Plain callables are wrapped automatically.
        unique:    If True, generated values must be unique within the table.
        nullable:  If True, some values may be None with probability `null_prob`.
        null_prob: Probability of generating None when `nullable=True`.
        transform: Optional post-processing function applied to non-null values.
    """
    generator: Union[Value[Any], Callable[[], Any]]
    unique: bool = False
    nullable: bool = False
    null_prob: float = 0.0
    transform: Optional[Callable[[Any], Any]] = None

    def normalized(self) -> "ColumnSpec":
        gen = self.generator
        if not isinstance(gen, Value) and callable(gen):
            gen = Lambda(gen)  # wrap plain callable
        return ColumnSpec(
            generator=gen,
            unique=self.unique,
            nullable=self.nullable,
            null_prob=self.null_prob,
            transform=self.transform,
        )


class ForeignKey(Value[Any]):
    """Lazy foreign key that samples values from another table's column.

    Use this in a ColumnSpec as `generator=ForeignKey('users', 'id')`.

    During DataBase.generate(), foreign keys are bound to the actual
    list of values from the referenced table, after that they behave
    like RandomValue over that list.
    """

    def __init__(self, table_name: str, column_name: str):
        super().__init__(None)
        self.table_name = table_name
        self.column_name = column_name
        self._bound: Optional[RandomValue[Any]] = None

    def bind(self, values: Iterable[Any]) -> None:
        self._bound = RandomValue(list(values))

    def __call__(self) -> Any:
        if self._bound is None:
            raise RuntimeError(
                f"ForeignKey({self.table_name}.{self.column_name}) is not bound yet"
            )
        return self._bound()


# ---- Table -------------------------------------------------------------------

class Table:
    """Represents a database table and can generate synthetic rows.

    Example:
        users = Table(
            name='users',
            rows=100,
            columns={
                'id': ColumnSpec(generator=lambda: rd.randint(1, 10_000), unique=True),
                'name': ColumnSpec(generator=lambda: f'user_{rd.randint(1, 9999)}'),
            },
        )

        posts = Table(
            name='posts',
            rows=200,
            columns={
                'id': ColumnSpec(generator=lambda: rd.randint(1, 10_000), unique=True),
                'user_id': ColumnSpec(generator=ForeignKey('users', 'id')),
                'title': ColumnSpec(generator=lambda: f'post_{rd.randint(1, 9999)}'),
            },
        )
    """

    def __init__(
            self,
            name: str,
            rows: Union[int, Value[int]],
            columns: Dict[str, Union[ColumnSpec, Value[Any], Callable[[], Any]]],
    ) -> None:
        self.name = name
        self._rows: Value[int] = rows if isinstance(rows, Value) else Value(rows)
        # normalize columns to ColumnSpec with Value-based generators
        self.columns: Dict[str, ColumnSpec] = {
            col: (spec.normalized() if isinstance(spec, ColumnSpec) else ColumnSpec(spec).normalized())
            for col, spec in columns.items()
        }
        self._data: List[Dict[str, Any]] = []  # generated rows cache

    # --- metadata & dependencies ---

    def depends_on(self) -> Set[str]:
        deps: Set[str] = set()
        for spec in self.columns.values():
            gen = spec.generator
            if isinstance(gen, ForeignKey):
                deps.add(gen.table_name)
        deps.discard(self.name)
        return deps

    # --- generation ---

    def generate(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            rd.seed(seed)
        self._data = []

        # Prepare uniqueness trackers
        unique_trackers: Dict[str, Set[Any]] = {
            col: set() for col, spec in self.columns.items() if spec.unique
        }

        n_rows = self._rows()
        for _ in range(n_rows):
            row: Dict[str, Any] = {}
            for col, spec in self.columns.items():
                value = None
                # Handle nullability
                if spec.nullable and spec.null_prob > 0 and rd.random() < spec.null_prob:
                    value = None
                else:
                    value = spec.generator()
                    if value is not None and spec.transform is not None:
                        value = spec.transform(value)

                # Enforce uniqueness with retry (bounded attempts)
                if col in unique_trackers and value is not None:
                    attempts = 0
                    while value in unique_trackers[col] and attempts < 1000:
                        value = spec.generator()
                        if value is not None and spec.transform is not None:
                            value = spec.transform(value)
                        attempts += 1
                    if value in unique_trackers[col]:
                        raise RuntimeError(
                            f"Failed to generate unique value for column '{col}' in table '{self.name}'"
                        )
                    unique_trackers[col].add(value)

                row[col] = value
            self._data.append(row)
        return self._data

    # --- access & export ---

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._data)

    def column(self, name: str) -> List[Any]:
        return [r.get(name) for r in self._data]

    def to_csv(self, file_path: str, header: bool = True) -> None:
        fieldnames = list(self.columns.keys())
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if header:
                writer.writeheader()
            for r in self._data:
                writer.writerow(r)


# ---- DataBase ----------------------------------------------------------------

class DataBase(Value[Dict[str, List[Dict[str, Any]]]]):
    """Holds multiple tables and generates them in dependency order.

    Supports ForeignKey dependencies between tables.
    After generation, the data can be accessed via `data['table_name']`.
    """

    def __init__(self, *tables: Table) -> None:
        super().__init__(None)
        by_name: Dict[str, Table] = {}
        for t in tables:
            if t.name in by_name:
                raise ValueError(f"Duplicate table name: {t.name}")
            by_name[t.name] = t
        self.tables: Dict[str, Table] = by_name
        self.data: Dict[str, List[Dict[str, Any]]] = {}

    # --- dependency resolution ---

    def _dependency_graph(self) -> Dict[str, Set[str]]:
        return {name: table.depends_on() for name, table in self.tables.items()}

    def _toposort(self) -> List[str]:
        deps = {k: set(v) for k, v in self._dependency_graph().items()}
        result: List[str] = []
        no_deps = [name for name, d in deps.items() if not d]
        while no_deps:
            name = no_deps.pop()
            result.append(name)
            # remove this as a dependency from others
            for other, d in deps.items():
                if name in d:
                    d.remove(name)
                    if not d and other not in result and other not in no_deps:
                        no_deps.append(other)
        # any remaining deps indicate a cycle or missing table
        remaining = {k: v for k, v in deps.items() if v and k not in result}
        if remaining:
            raise RuntimeError(f"Unresolvable dependencies: {remaining}")
        # include any isolated already-added tables and maintain order
        return result

    # --- generation orchestrator ---

    def generate(self, seed: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        if seed is not None:
            rd.seed(seed)
        self.data = {}
        order = self._toposort()
        # bind foreign keys progressively, then generate
        for name in order:
            tbl = self.tables[name]
            # try to bind any ForeignKey columns that target already generated tables
            for col, spec in tbl.columns.items():
                gen = spec.generator
                if isinstance(gen, ForeignKey):
                    ref_table = gen.table_name
                    ref_col = gen.column_name
                    if ref_table not in self.data:
                        # not yet generated, will be bound after generating the referenced table
                        continue
                    gen.bind([row[ref_col] for row in self.data[ref_table]])

            self.data[name] = tbl.generate()

            # After generation, bind FKs in tables that depend on this one
            for other in self.tables.values():
                for col, spec in other.columns.items():
                    g = spec.generator
                    if isinstance(g, ForeignKey) and g.table_name == name:
                        g.bind([row[g.column_name] for row in self.data[name]])

        return self.data

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate and return the database data to comply with Value interface."""
        return self.generate()

    # convenience accessors
    def table(self, name: str) -> Table:
        return self.tables[name]

    def rows(self, name: str) -> List[Dict[str, Any]]:
        return self.data.get(name, [])
