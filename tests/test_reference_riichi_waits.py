"""Independent tests for the reference-only 13-tile wait finder."""

import ast
from collections.abc import Iterable
from importlib.util import resolve_name
from pathlib import Path

import pytest

from mahjong_analysis.riichi_wait_reference import (
    REFERENCE_TILE_KINDS,
    ReferenceHandWaits,
    ReferenceMeld,
    ReferenceWaitDetail,
    calculate_reference_hand_waits,
    normalize_reference_tile,
    reference_tile_to_index,
)

DetailTuple = tuple[str, str, str]


def _tiles(*groups: str) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        if group in {"5mr", "5pr", "5sr"}:
            result.append(group)
        elif group[-1:] in {"m", "p", "s"}:
            suit = group[-1]
            result.extend(f"{rank}{suit}" for rank in group[:-1])
        else:
            result.extend(group)
    return tuple(result)


def _detail_tuples(waits: ReferenceHandWaits) -> tuple[DetailTuple, ...]:
    return tuple(
        (detail.wait_tile, detail.hand_type, detail.wait_shape)
        for detail in waits.wait_details
    )


REFERENCE_CASES = (
    pytest.param(
        _tiles("123m", "123p", "789p", "EE", "45s"),
        (),
        ("3s", "6s"),
        2,
        (
            ("3s", "standard", "ryanmen"),
            ("6s", "standard", "ryanmen"),
        ),
        ("ryanmen",),
        True,
        True,
        False,
        id="R1-pure-ryanmen",
    ),
    pytest.param(
        _tiles("123m", "123p", "789p", "EE", "46s"),
        (),
        ("5s",),
        1,
        (("5s", "standard", "kanchan"),),
        ("kanchan",),
        False,
        False,
        False,
        id="R2-kanchan",
    ),
    pytest.param(
        _tiles("123m", "456m", "789p", "EE", "12s"),
        (),
        ("3s",),
        1,
        (("3s", "standard", "penchan"),),
        ("penchan",),
        False,
        False,
        False,
        id="R3-penchan",
    ),
    pytest.param(
        _tiles("123m", "456m", "789m", "55p", "77s"),
        (),
        ("5p", "7s"),
        2,
        (
            ("5p", "standard", "shanpon"),
            ("7s", "standard", "shanpon"),
        ),
        ("shanpon",),
        False,
        False,
        False,
        id="R4-shanpon",
    ),
    pytest.param(
        _tiles("123m", "456m", "789m", "123p", "5s"),
        (),
        ("5s",),
        1,
        (("5s", "standard", "tanki"),),
        ("tanki",),
        False,
        False,
        False,
        id="R5-tanki",
    ),
    pytest.param(
        _tiles("34567m", "123p", "789p", "EE"),
        (),
        ("2m", "5m", "8m"),
        3,
        (
            ("2m", "standard", "ryanmen"),
            ("5m", "standard", "ryanmen"),
            ("8m", "standard", "ryanmen"),
        ),
        ("ryanmen",),
        True,
        False,
        True,
        id="R6-three-sided-ryanmen",
    ),
    pytest.param(
        _tiles("2345678m", "123p", "789p"),
        (),
        ("2m", "5m", "8m"),
        3,
        (
            ("2m", "standard", "tanki"),
            ("5m", "standard", "tanki"),
            ("8m", "standard", "tanki"),
        ),
        ("tanki",),
        False,
        False,
        True,
        id="R7-multiwait-without-ryanmen",
    ),
    pytest.param(
        _tiles("2333456m", "123p", "789p"),
        (),
        ("1m", "2m", "4m", "7m"),
        4,
        (
            ("1m", "standard", "ryanmen"),
            ("2m", "standard", "tanki"),
            ("4m", "standard", "ryanmen"),
            ("7m", "standard", "ryanmen"),
        ),
        ("ryanmen", "tanki"),
        True,
        False,
        True,
        id="R8-compound-with-ryanmen",
    ),
    pytest.param(
        _tiles("123m", "456m", "789m", "4556p"),
        (),
        ("5p",),
        1,
        (
            ("5p", "standard", "kanchan"),
            ("5p", "standard", "tanki"),
        ),
        ("kanchan", "tanki"),
        False,
        False,
        False,
        id="R9-same-tile-multiple-shapes",
    ),
    pytest.param(
        _tiles("11m", "22m", "33p", "44p", "55s", "66s", "E"),
        (),
        ("E",),
        1,
        (("E", "chiitoitsu", "tanki"),),
        ("tanki",),
        False,
        False,
        False,
        id="R10-chiitoitsu",
    ),
    pytest.param(
        _tiles("19m", "19p", "19s", "EE", "S", "W", "N", "P", "F"),
        (),
        ("C",),
        1,
        (("C", "kokushi", "kokushi_single"),),
        ("kokushi_single",),
        False,
        False,
        False,
        id="R11-kokushi-single",
    ),
    pytest.param(
        _tiles("19m", "19p", "19s", "ESWNPFC"),
        (),
        ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"),
        13,
        tuple(
            (tile, "kokushi", "kokushi_13men")
            for tile in (
                "1m",
                "9m",
                "1p",
                "9p",
                "1s",
                "9s",
                "E",
                "S",
                "W",
                "N",
                "P",
                "F",
                "C",
            )
        ),
        ("kokushi_13men",),
        False,
        False,
        True,
        id="R12-kokushi-thirteen-sided",
    ),
    pytest.param(
        _tiles("123m", "123p", "789p", "EE", "4s", "5sr"),
        (),
        ("3s", "6s"),
        2,
        (
            ("3s", "standard", "ryanmen"),
            ("6s", "standard", "ryanmen"),
        ),
        ("ryanmen",),
        True,
        True,
        False,
        id="R13-red-five",
    ),
    pytest.param(
        _tiles("123p", "789p", "EE", "45s"),
        (ReferenceMeld(_tiles("9999m")),),
        ("3s", "6s"),
        2,
        (
            ("3s", "standard", "ryanmen"),
            ("6s", "standard", "ryanmen"),
        ),
        ("ryanmen",),
        True,
        True,
        False,
        id="R14-fixed-ankan",
    ),
    pytest.param(
        _tiles("22234567m", "22p", "789s"),
        (),
        ("2m", "5m", "8m", "2p"),
        4,
        (
            ("2m", "standard", "ryanmen"),
            ("2m", "standard", "shanpon"),
            ("5m", "standard", "ryanmen"),
            ("8m", "standard", "ryanmen"),
            ("2p", "standard", "shanpon"),
        ),
        ("ryanmen", "shanpon"),
        True,
        False,
        True,
        id="R15-compound-same-tile-two-shapes",
    ),
    pytest.param(
        _tiles("11122233m", "456p", "EE"),
        (),
        ("3m", "E"),
        2,
        (
            ("3m", "standard", "penchan"),
            ("3m", "standard", "shanpon"),
            ("E", "standard", "shanpon"),
        ),
        ("penchan", "shanpon"),
        False,
        False,
        False,
        id="R16-distinct-completions",
    ),
)


@pytest.mark.parametrize(
    (
        "concealed_tiles",
        "fixed_melds",
        "expected_wait_tiles",
        "expected_wait_tile_count",
        "expected_wait_details",
        "expected_wait_shapes",
        "expected_contains_ryanmen",
        "expected_is_pure_ryanmen",
        "expected_is_multiwait",
    ),
    REFERENCE_CASES,
)
def test_reference_waits_r1_through_r16(
    concealed_tiles: tuple[str, ...],
    fixed_melds: tuple[ReferenceMeld, ...],
    expected_wait_tiles: tuple[str, ...],
    expected_wait_tile_count: int,
    expected_wait_details: tuple[DetailTuple, ...],
    expected_wait_shapes: tuple[str, ...],
    expected_contains_ryanmen: bool,
    expected_is_pure_ryanmen: bool,
    expected_is_multiwait: bool,
) -> None:
    waits = calculate_reference_hand_waits(concealed_tiles, fixed_melds)

    assert waits.wait_tiles == expected_wait_tiles
    assert waits.wait_tile_count == expected_wait_tile_count
    assert _detail_tuples(waits) == expected_wait_details
    assert waits.wait_shapes == expected_wait_shapes
    assert waits.contains_ryanmen is expected_contains_ryanmen
    assert waits.is_pure_ryanmen is expected_is_pure_ryanmen
    assert waits.is_multiwait is expected_is_multiwait


def test_reference_counts_completed_honor_triplet_in_residue() -> None:
    waits = calculate_reference_hand_waits(_tiles("EEE", "123m", "456p", "NN", "45s"))

    assert waits.wait_tiles == ("3s", "6s")
    assert _detail_tuples(waits) == (
        ("3s", "standard", "ryanmen"),
        ("6s", "standard", "ryanmen"),
    )


def test_reference_retains_standard_and_chiitoitsu_interpretations() -> None:
    waits = calculate_reference_hand_waits(_tiles("112233m", "445566p", "E"))

    assert waits.wait_tiles == ("E",)
    assert waits.wait_tile_count == 1
    assert _detail_tuples(waits) == (
        ("E", "standard", "tanki"),
        ("E", "chiitoitsu", "tanki"),
    )
    assert waits.wait_shapes == ("tanki",)
    assert waits.contains_ryanmen is False
    assert waits.is_pure_ryanmen is False
    assert waits.is_multiwait is False


@pytest.mark.parametrize(
    "concealed_tiles",
    (
        _tiles("1111m", "22p", "33p", "44s", "55s", "E"),
        _tiles("19m", "19p", "1s", "EE", "SS", "W", "N", "P", "F"),
    ),
    ids=("chiitoitsu-four-copies", "kokushi-two-missing"),
)
def test_reference_special_hand_negative(
    concealed_tiles: tuple[str, ...],
) -> None:
    waits = calculate_reference_hand_waits(concealed_tiles)

    assert waits.wait_tiles == ()
    assert waits.wait_details == ()


def test_reference_normalizes_red_five_pair_wait() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("123p", "456p", "789p", "EE", "5m", "5mr")
    )

    assert waits.wait_tiles == ("5m", "E")
    assert _detail_tuples(waits) == (
        ("5m", "standard", "shanpon"),
        ("E", "standard", "shanpon"),
    )
    assert "5mr" not in waits.wait_tiles


def test_reference_excludes_wait_kind_already_owned_four_times() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("123m", "456p", "EE", "12s"),
        (ReferenceMeld(_tiles("3333s")),),
    )

    assert waits.wait_tiles == ()
    assert waits.wait_details == ()


def test_reference_red_ankan_excludes_one_side_of_ryanmen() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("34m", "123p", "789p", "EE"),
        (ReferenceMeld(("5m", "5m", "5m", "5mr")),),
    )

    assert waits.wait_tiles == ("2m",)
    assert waits.wait_tile_count == 1
    assert _detail_tuples(waits) == (("2m", "standard", "ryanmen"),)
    assert waits.wait_shapes == ("ryanmen",)
    assert waits.contains_ryanmen is True
    assert waits.is_pure_ryanmen is False
    assert waits.is_multiwait is False


@pytest.mark.parametrize(
    ("fragment", "expected_wait_tiles", "expected_wait_details"),
    (
        (
            "89s",
            ("7s",),
            (("7s", "standard", "penchan"),),
        ),
        (
            "78s",
            ("6s", "9s"),
            (
                ("6s", "standard", "ryanmen"),
                ("9s", "standard", "ryanmen"),
            ),
        ),
    ),
    ids=("upper-end-penchan", "upper-end-ryanmen"),
)
def test_reference_upper_end_fragments(
    fragment: str,
    expected_wait_tiles: tuple[str, ...],
    expected_wait_details: tuple[DetailTuple, ...],
) -> None:
    waits = calculate_reference_hand_waits(
        _tiles("123m", "123p", "789p", "EE", fragment)
    )

    assert waits.wait_tiles == expected_wait_tiles
    assert _detail_tuples(waits) == expected_wait_details


def test_reference_single_fixed_ankan_with_raw_red_five() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("123m", "456p", "EE", "78s"),
        (ReferenceMeld(("5m", "5m", "5m", "5mr")),),
    )

    assert waits.wait_tiles == ("6s", "9s")
    assert _detail_tuples(waits) == (
        ("6s", "standard", "ryanmen"),
        ("9s", "standard", "ryanmen"),
    )


def test_reference_multiple_ankan() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("123p", "EE", "45s"),
        (
            ReferenceMeld(_tiles("1111m")),
            ReferenceMeld(_tiles("9999m")),
        ),
    )

    assert waits.wait_tiles == ("3s", "6s")
    assert _detail_tuples(waits) == (
        ("3s", "standard", "ryanmen"),
        ("6s", "standard", "ryanmen"),
    )


def test_reference_three_ankan_fragment_has_zero_meld_residue() -> None:
    waits = calculate_reference_hand_waits(
        _tiles("EE", "12s"),
        (
            ReferenceMeld(_tiles("1111m")),
            ReferenceMeld(_tiles("9999m")),
            ReferenceMeld(_tiles("1111p")),
        ),
    )

    assert waits.wait_tiles == ("3s",)
    assert waits.wait_tile_count == 1
    assert _detail_tuples(waits) == (("3s", "standard", "penchan"),)
    assert waits.wait_shapes == ("penchan",)
    assert waits.contains_ryanmen is False
    assert waits.is_pure_ryanmen is False
    assert waits.is_multiwait is False


def test_reference_zero_required_melds_four_ankan_tanki() -> None:
    waits = calculate_reference_hand_waits(
        ("E",),
        (
            ReferenceMeld(_tiles("1111m")),
            ReferenceMeld(_tiles("9999m")),
            ReferenceMeld(_tiles("1111p")),
            ReferenceMeld(_tiles("9999p")),
        ),
    )

    assert waits.wait_tiles == ("E",)
    assert _detail_tuples(waits) == (("E", "standard", "tanki"),)


def test_reference_deduplicates_identical_details() -> None:
    waits = calculate_reference_hand_waits(_tiles("34567m", "123p", "789p", "EE"))

    assert _detail_tuples(waits).count(("5m", "standard", "ryanmen")) == 1
    assert len(waits.wait_details) == 3


def test_reference_result_is_independent_of_concealed_input_order() -> None:
    concealed = _tiles("22234567m", "22p", "789s")

    forward = calculate_reference_hand_waits(concealed)
    reversed_input = calculate_reference_hand_waits(tuple(reversed(concealed)))

    assert reversed_input == forward


@pytest.mark.parametrize(
    "bad_tiles",
    (
        ("0m",) + ("1p",) * 12,
        (object(),) + ("1p",) * 12,
    ),
    ids=("unknown-string", "non-string"),
)
def test_reference_rejects_invalid_concealed_tile(
    bad_tiles: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError), match="tile"):
        calculate_reference_hand_waits(bad_tiles)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ReferenceMeld(("1m", "1m", "1m")),
        lambda: ReferenceMeld(("1m", "1m", "1m", "2m")),
        lambda: ReferenceMeld(("1m", "1m", "1m", "1m"), "pon"),
        lambda: ReferenceMeld(["1m", "1m", "1m", "1m"]),
    ),
    ids=("three-tiles", "mixed-kind", "unsupported-type", "mutable-tiles"),
)
def test_reference_rejects_invalid_fixed_meld(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_reference_models_reject_noncanonical_wait_values() -> None:
    with pytest.raises(ValueError, match="normalized"):
        ReferenceWaitDetail("5mr", "standard", "shanpon")
    with pytest.raises(ValueError, match="combination"):
        ReferenceWaitDetail("5m", "chiitoitsu", "ryanmen")
    with pytest.raises(TypeError, match="tuple"):
        ReferenceHandWaits([], ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="34-kind order"):
        ReferenceHandWaits(
            ("6s", "3s"),
            (
                ReferenceWaitDetail("3s", "standard", "ryanmen"),
                ReferenceWaitDetail("6s", "standard", "ryanmen"),
            ),
        )


def test_reference_model_rejects_duplicate_details() -> None:
    detail = ReferenceWaitDetail("3s", "standard", "ryanmen")

    with pytest.raises(ValueError, match="wait_details must not contain duplicates"):
        ReferenceHandWaits(("3s",), (detail, detail))


def test_reference_model_rejects_detail_tile_set_mismatch() -> None:
    detail = ReferenceWaitDetail("3s", "standard", "ryanmen")

    with pytest.raises(ValueError, match="wait_tiles must match tiles in wait_details"):
        ReferenceHandWaits(("6s",), (detail,))


def test_reference_model_rejects_nested_mutable_detail() -> None:
    detail = ReferenceWaitDetail("3s", "standard", "ryanmen")

    with pytest.raises(TypeError, match="ReferenceWaitDetail instances"):
        ReferenceHandWaits(("3s",), ([detail],))  # type: ignore[arg-type]


def test_reference_empty_model_has_no_waits_or_positive_flags() -> None:
    waits = ReferenceHandWaits((), ())

    assert waits.wait_tiles == ()
    assert waits.wait_details == ()
    assert waits.wait_tile_count == 0
    assert waits.wait_shapes == ()
    assert waits.contains_ryanmen is False
    assert waits.is_pure_ryanmen is False
    assert waits.is_multiwait is False


def test_reference_tile_order_and_red_normalization_are_self_contained() -> None:
    assert len(REFERENCE_TILE_KINDS) == 34
    assert len(set(REFERENCE_TILE_KINDS)) == 34
    assert normalize_reference_tile("5mr") == "5m"
    assert normalize_reference_tile("5pr") == "5p"
    assert normalize_reference_tile("5sr") == "5s"
    assert tuple(map(reference_tile_to_index, REFERENCE_TILE_KINDS)) == tuple(range(34))


_FORBIDDEN_PRODUCTION_MODULES = (
    "mahjong_analysis.riichi",
    "mahjong_analysis.hand_waits",
    "mahjong_analysis.riichi_wait_records",
    "mahjong_analysis.tiles",
    "mahjong_analysis.mjai",
)


def _referenced_imports(source: str, *, package: str) -> set[str]:
    """Resolve static imports without importing or executing the scanned code."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            imported.add(module)
            # A from-import may refer to a child module rather than an attribute.
            # Check both names conservatively; aliases do not change the target.
            imported.update(
                f"{module}.{alias.name}" for alias in node.names if alias.name != "*"
            )
    return imported


def _forbidden_imports(source: str, *, package: str) -> set[str]:
    return {
        module
        for module in _referenced_imports(source, package=package)
        if any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for forbidden in _FORBIDDEN_PRODUCTION_MODULES
        )
        or module.split(".")[0] in {"tests", "conftest"}
        or module.split(".")[0].startswith("test_")
    }


def _file_import_violations(
    paths: Iterable[Path], *, source_root: Path
) -> dict[Path, set[str]]:
    violations: dict[Path, set[str]] = {}
    for path in sorted(paths):
        package = ".".join(path.relative_to(source_root).parent.parts)
        forbidden = _forbidden_imports(
            path.read_text(encoding="utf-8"), package=package
        )
        if forbidden:
            violations[path] = forbidden
    return violations


def _reference_package_import_violations(
    package_root: Path, *, source_root: Path
) -> dict[Path, set[str]]:
    return _file_import_violations(package_root.rglob("*.py"), source_root=source_root)


@pytest.mark.parametrize(
    ("source", "expected_module"),
    (
        ("import mahjong_analysis.riichi", "mahjong_analysis.riichi"),
        ("import mahjong_analysis.tiles as tiles", "mahjong_analysis.tiles"),
        (
            "from mahjong_analysis.hand_waits import calculate_hand_waits",
            "mahjong_analysis.hand_waits",
        ),
        (
            "from mahjong_analysis.hand_waits import calculate_hand_waits as calc",
            "mahjong_analysis.hand_waits",
        ),
        (
            "from mahjong_analysis import hand_waits",
            "mahjong_analysis.hand_waits",
        ),
        (
            "from mahjong_analysis import hand_waits as hw",
            "mahjong_analysis.hand_waits",
        ),
        (
            "import mahjong_analysis.riichi_wait_records.helpers as helpers",
            "mahjong_analysis.riichi_wait_records.helpers",
        ),
        (
            "from mahjong_analysis.mjai.reader import load",
            "mahjong_analysis.mjai.reader",
        ),
        (
            "from ..hand_waits import calculate_hand_waits as calc",
            "mahjong_analysis.hand_waits",
        ),
        ("from .. import tiles", "mahjong_analysis.tiles"),
        ("from .. import mjai as reader", "mahjong_analysis.mjai"),
        ("from ..riichi.helpers import helper", "mahjong_analysis.riichi.helpers"),
        (
            "from test_hand_waits import KOKUSHI_TILES",
            "test_hand_waits",
        ),
        (
            "from tests.test_hand_waits import KOKUSHI_TILES",
            "tests.test_hand_waits",
        ),
        ("from tests.some_helper import fixture", "tests.some_helper"),
        ("from tests import some_helper as fixture", "tests.some_helper"),
        ("import test_hand_waits as fixtures", "test_hand_waits"),
        ("from conftest import fixture", "conftest"),
    ),
)
def test_import_guard_detects_forbidden_source(
    source: str, expected_module: str
) -> None:
    forbidden = _forbidden_imports(
        source, package="mahjong_analysis.riichi_wait_reference"
    )

    assert expected_module in forbidden


@pytest.mark.parametrize(
    "source",
    (
        "import ast",
        "import collections as containers",
        "from collections.abc import Iterable as Items",
        "from dataclasses import dataclass",
        "from pathlib import Path",
        "from typing import Literal",
        "from .model import ReferenceMeld",
        "from . import waits",
        "from mahjong_analysis import riichi_wait_reference as reference",
    ),
)
def test_import_guard_allows_standard_library_and_reference_source(source: str) -> None:
    assert (
        _forbidden_imports(source, package="mahjong_analysis.riichi_wait_reference")
        == set()
    )


def test_import_guard_recursively_resolves_nested_package_imports(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "mahjong_analysis" / "riichi_wait_reference"
    nested_root = package_root / "nested"
    nested_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        "from .model import ReferenceMeld\n", encoding="utf-8"
    )
    nested_init = nested_root / "__init__.py"
    nested_init.write_text("from ... import tiles\n", encoding="utf-8")
    nested_module = nested_root / "helper.py"
    nested_module.write_text(
        "from ...hand_waits import calculate_hand_waits\n", encoding="utf-8"
    )

    violations = _reference_package_import_violations(
        package_root, source_root=tmp_path
    )

    assert set(violations) == {nested_init, nested_module}
    assert "mahjong_analysis.tiles" in violations[nested_init]
    assert "mahjong_analysis.hand_waits" in violations[nested_module]


def test_reference_package_does_not_import_production_modules_or_fixtures() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    package_root = source_root / "mahjong_analysis" / "riichi_wait_reference"
    reference_tests = (
        Path(__file__).resolve(),
        repository_root / "tests/test_reference_riichi_mjai.py",
    )

    assert (package_root / "__init__.py").is_file()
    assert (
        _reference_package_import_violations(package_root, source_root=source_root)
        == {}
    )
    assert _file_import_violations(reference_tests, source_root=repository_root) == {}
