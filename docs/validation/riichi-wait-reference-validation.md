# Issue #24 成立リーチ待ち抽出の独立reference検証記録

- 初回記録日: 2026-09-10
- post-fix再検証追記日: 2026-09-13
- canonical full export追記日: 2026-09-14
- 対象仕様: [成立リーチの待ち形・両面判定・巡目抽出仕様](../specs/riichi-wait-extraction.md)
- 対象: productionの成立リーチ抽出・手牌再生・待ち判定と、独立reference実装
- post-fix対象commit: `e596560 Handle Tenhou fifth-tile waits`
- canonical generator commit: `6904f8661bd414cce081ac524e031e38821a0284`

## 結論

5枚目待ち修正後のcommit `e596560`について正式比較を再実行した。重複を除く
post-fix検証範囲は355,962ファイル、355,961対象対局、2,047,821東場、
1,585,473成立リーチであり、production/referenceの全difference classは0だった。

このpost-fix範囲には、2009年全件、2023年全件、2025年全件、および2023年を除く
2010〜2024各年の決定論的先頭100ファイルを含む。2009〜2025の全年全件比較ではない。

この再検証後、commit `6904f8661bd414cce081ac524e031e38821a0284`のclean worktreeから
2009〜2025のcanonical full exportを新規生成した。10,706,714成立リーチを同数の
レコードとして出力し、completion marker、manifest totals、全17年度artifactの
compressed sizeとSHA256を検証した。`data/processed/riichi-waits-v1`をschema v1の
canonical datasetとして確定した。

本書では、修正前の正式比較を履歴として残し、その後に共有仕様バグの発見・影響診断、
post-fix正式再検証を分けて記録する。独立アルゴリズム同士の一致でも、両者が共有する
仕様の誤りは検出できない。

これは独立実装との比較によって実装の信頼性を高める検証結果であり、実装の
正しさを証明するものではない。未検証の入力や不具合が理論上存在しないことも
意味しない。

## 比較条件

productionとreferenceには同一の入力ファイル集合を渡し、両経路が対象対局判定、
MJAI読込・状態再生、成立リーチ検出、待ち判定、派生値計算をそれぞれ行った。
referenceはproductionの主要モジュールをimportせず、待ち判定も13枚手牌から
未完成断片を予約する独立方式を使用した。

候補は仕様で定めた
`(relative_source_path, start_kyoku_line, reach_line)` で対応付け、次の差分分類を
すべて検査した。

- `production_only`
- `reference_only`
- `field_mismatch`
- `scope_mismatch`
- `processing_error`

正式比較項目にはactor、宣言牌、本人打牌数、宣言直後手牌、fixed melds、順序を
保持した河履歴、`wait_tiles`、`wait_details`および全派生値を含む。いずれかの
差分分類が1件でも残る場合はPASSとしない。

決定論的Nファイル検証では、指定年のraw root配下の `.mjson` を列挙し、raw root
からの相対パスを `/` 区切りへ正規化して辞書順に並べ、対象対局判定前の先頭N件を
固定した。productionとreferenceには完全に同じ集合を渡した。

## 1. 5枚目待ち修正前のvalidation履歴

以下は、天鳳の5枚目待ちに関する共有仕様バグを修正する前の正式比較結果である。
現在のコードに対するpost-fix再検証結果ではなく、検証の履歴として保持する。

### 段階的検証

| 段階 | 範囲 | 結果 | 備考 |
|---|---:|---:|---|
| 人工待ちケース | R1〜R16 | PASS | 待ち情報7項目を期待値と完全一致 |
| reference waits独立大量生成診断 | 13,296牌姿 | 差分0 | 暗槓0〜4、七対子、国士、赤5、空待ち、2牌断片595通り等を含む |
| 目視監査済み2025年 | 8ファイル、45東場、30候補 | 差分0 / PASS | 元MJAIを追跡可能な監査対象 |
| 2025年決定論的100ファイル | 100ファイル、593東場、418候補 | 差分0 / PASS | 2025年全件に包含 |
| 2025年決定論的1000ファイル | 1,000ファイル、5,744東場、4,384候補 | 差分0 / PASS | 2025年全件に包含、実行時間1分58.8秒 |
| 2025年全件 | 178,888ファイル | 差分0 / PASS | 詳細は後述 |
| 2009年全件 | 6,897ファイル | 差分0 / PASS | 詳細は後述 |
| 2010〜2024 | 各年決定論的100ファイル | 差分0 / PASS | 各年全件ではない |

8ファイル、100ファイル、1000ファイルの検証結果は段階的な回帰・監査記録として
残すが、すべて2025年全件に含まれるため、正式検証範囲の総数には重複加算しない。
人工ケースと独立大量生成診断も実牌譜のファイル・東場・候補総数には加算しない。

### 2025年全件

- `scanned_files`: 178,888
- `production_target_games`: 178,887
- `reference_target_games`: 178,887
- `production_east_kyokus`: 1,028,072
- `reference_east_kyokus`: 1,028,072
- `production_candidates`: 794,774
- `reference_candidates`: 794,774
- `production_only`: 0
- `reference_only`: 0
- `field_mismatch`: 0
- `scope_mismatch`: 0
- `processing_error`: 0
- `result`: PASS
- `ExitCode`: 0
- `Elapsed`: 02:26:53.6449482

走査ファイル178,888件のうち対象対局は両経路とも178,887件であり、対象範囲の
判定結果にも差分はなかった。

### 2009年全件

- `scanned_files`: 6,897
- `production_target_games` / `reference_target_games`: 6,897 / 6,897
- `production_east_kyokus` / `reference_east_kyokus`: 39,778 / 39,778
- `production_candidates` / `reference_candidates`: 28,298 / 28,298
- `production_only`: 0
- `reference_only`: 0
- `field_mismatch`: 0
- `scope_mismatch`: 0
- `processing_error`: 0
- `result`: PASS
- `ExitCode`: 0
- `Elapsed`: 00:05:32.5081940

### 2010〜2024各年の決定論的100ファイル

この検証は各年度の全件比較ではない。各年について、対象対局判定前に固定した
先頭100ファイルだけをproduction/referenceへ渡した。

| 年 | 入力ファイル | production candidates | reference candidates | 全difference class | 結果 |
|---:|---:|---:|---:|---:|---:|
| 2010 | 100 | 374 | 374 | 0 | PASS |
| 2011 | 100 | 395 | 395 | 0 | PASS |
| 2012 | 100 | 381 | 381 | 0 | PASS |
| 2013 | 100 | 397 | 397 | 0 | PASS |
| 2014 | 100 | 397 | 397 | 0 | PASS |
| 2015 | 100 | 419 | 419 | 0 | PASS |
| 2016 | 100 | 425 | 425 | 0 | PASS |
| 2017 | 100 | 426 | 426 | 0 | PASS |
| 2018 | 100 | 392 | 392 | 0 | PASS |
| 2019 | 100 | 440 | 440 | 0 | PASS |
| 2020 | 100 | 441 | 441 | 0 | PASS |
| 2021 | 100 | 406 | 406 | 0 | PASS |
| 2022 | 100 | 411 | 411 | 0 | PASS |
| 2023 | 100 | 465 | 465 | 0 | PASS |
| 2024 | 100 | 454 | 454 | 0 | PASS |
| **合計** | **1,500** | **6,223** | **6,223** | **0** | **PASS** |

15年度合計の東場は8,560局である。`production_only`、`reference_only`、
`field_mismatch`、`scope_mismatch`、`processing_error` は全年度ですべて0だった。

### 重複を除く正式検証範囲

総数は、2025年全件、2009年全件、2010〜2024各年100ファイルだけを加算した。

- files: `178,888 + 6,897 + 1,500 = 187,285`
- east kyokus: `1,028,072 + 39,778 + 8,560 = 1,076,410`
- production candidates: `794,774 + 28,298 + 6,223 = 829,295`
- reference candidates: `794,774 + 28,298 + 6,223 = 829,295`
- production/reference差分: 0

### 全年全件比較への拡張判断

2025年全件は2時間26分53.6449482秒、2009年全件は5分32.5081940秒を要した。
当時の仕様上の必須範囲である両年全件と2010〜2024各年100ファイルの比較は完了した。
2010〜2024の各年全件比較は必須完了条件ではなく、pre-fix validation時点では
実施していない。2025年全件だけでも複数時間を要すること、および必須範囲で差分が
検出されなかったことから、追加の2009〜2025年全件比較は別途必要性と実行資源を
判断して行う。

### 制約と未検証範囲

- 2010〜2024は各年100ファイルのみで、各年全件は未検証である
- 2009〜2025の全年度全件比較は実施していない
- 外部mahjongライブラリによる第三者照合は依存へ追加していない
- 差分0は指定入力と比較項目に対する結果であり、将来の形式差や未知の不正入力を
  網羅したことを意味しない

## 2. 天鳳の5枚目待ちに関する共有仕様バグと影響診断

[天鳳公式マニュアル](https://tenhou.net/man/index.html)では純手牌内で同牌種を4枚
使用していなければ5枚目待ちの聴牌を認める。両実装とも候補除外にfixed meldの
枚数まで合算していたため、独立比較はこの誤前提を検出できなかった。
修正仕様とR17〜R20は[抽出仕様](../specs/riichi-wait-extraction.md)を参照。

### 問題の実牌譜による回帰確認

`2023/2023103019gm-00a9-0000-4b80c963.mjson`の正式CLI比較を修正後に実施した。

- candidate key: `(2023/2023103019gm-00a9-0000-4b80c963.mjson, 339, 450)`
- actor 0、本人第13打、宣言牌`6m`（物理行451）、成立確認は物理行452
- 純手牌`44p 567p 12s 456s`、fixed ankan `3333s`
- `wait_tiles = (3s,)`、`wait_details = (3s/standard/penchan,)`
- ファイル全体: 1対象対局、5東場、production/reference各2候補
- 全5差分分類が0、`processing_error=0`、PASS（終了コード0）

### 既存年度出力のread-only影響診断

未完了full exportの`data/processed/riichi-waits-v1/2009.jsonl.gz`〜
`2022.jsonl.gz`だけを読み取り、8,329,334レコードを走査した。fixed meldありの
113,024レコードについて保存済みの宣言直後手牌から新仕様の待ちを再計算した。
rawの全件走査・再export、`.part`の復旧、artifactの削除・移動は行っていない。

分類方法（待ち探索本体に旧仕様分岐は追加しない）:

1. 新仕様の全待ちを`new_waits`とする。
2. 入力のconcealed + fixedの正規化所有枚数を数え、その枚数が4未満の牌だけを
   `new_waits`から残したものを`legacy_waits`とする。
3. `new_waits - legacy_waits`が空でなければ影響あり。
   - `legacy_waits`が空: 旧コードの空待ち停止ケース
   - `legacy_waits`が非空: 旧コードのsilent omission（一部欠落）ケース
4. 全113,024レコードで`legacy_waits`と保存済み`wait_tiles`の完全一致を確認した。
   影響ありの249レコードではreferenceでも再計算し、新仕様の全待ち・detailが
   productionと一致することを確認した。

| 年 | silent omission件数 |
|---|---:|
| 2009 | 2 |
| 2010 | 13 |
| 2011 | 6 |
| 2012 | 13 |
| 2013 | 21 |
| 2014 | 17 |
| 2015 | 18 |
| 2016 | 21 |
| 2017 | 22 |
| 2018 | 24 |
| 2019 | 24 |
| 2020 | 35 |
| 2021 | 13 |
| 2022 | 20 |
| 合計 | 249 |

同249件のうち`is_pure_ryanmen`が変わるものは204件、`is_multiwait`が変わるものは
71件（重複あり）。これら以外の派生値が不変であると主張する集計ではない。
保存済みレコード内の旧空待ち停止相当は0件だが、旧処理は空待ちを保存する前に
停止するため、この0件だけで未出力入力の不存在を主張できない。別途、上記2023年の
実牌譜で旧空待ち停止1件を確認した。2023年途中出力・2024/2025年の影響総数は未調査。

一部欠落の2実例をrawからも再生し、production/referenceの全候補が一致した。

| source（年/filename） | start / reach行 | actor | 旧待ち | 修正後待ち |
|---|---|---:|---|---|
| `2009/2009120802gm-00a9-0000-2624052b.mjson` | 600 / 716 | 2 | `6s` | `6s, 9s` |
| `2009/2009123103gm-00a9-0000-d8f3d7d5.mjson` | 84 / 147 | 1 | `5m, 8m` | `2m, 5m, 8m` |

旧仕様で出力済みの年度は修正後canonical datasetへ流用せず、旧checkpointから
resumeしなかった。failed datasetは
`data/processed/riichi-waits-v1-pre-fifth-tile-fix-failed`へ退避し、修正後の
canonical datasetと分離した。canonical full exportはcommit `e596560`を含む
`6904f8661bd414cce081ac524e031e38821a0284`のclean worktreeから、cleanなoutput rootへ
2009〜2025を最初から生成した。

## 3. e596560以降のpost-fix formal revalidation

5枚目待ち修正後のproduction/referenceを、commit
`e596560 Handle Tenhou fifth-tile waits`で正式に再比較した。各検証で
`production_only`、`reference_only`、`field_mismatch`、`scope_mismatch`、
`processing_error`はすべて0であり、終了コード0でPASSした。
R1〜R16は既存の期待値を変えず、5枚目待ちのR17〜R20とともに修正後コードの
production/reference人工テストで完全一致を確認した。

### 2009年全件

- `scanned_files`: 6,897
- `production_target_games`: 6,897
- `reference_target_games`: 6,897
- `production_east_kyokus`: 39,778
- `reference_east_kyokus`: 39,778
- `production_candidates`: 28,298
- `reference_candidates`: 28,298
- 全difference class: 0
- `result`: PASS
- `ExitCode`: 0
- `Elapsed`: 00:05:07.0824648

### 2010〜2024各年の決定論的先頭100ファイル

各年について、raw rootからの相対パスを`/`区切りへ正規化して辞書順に並べ、
対象対局判定前の先頭100ファイルをproduction/referenceへ同一に渡した。全15年で
全difference classが0となり、PASSした。この検証は各年全件比較ではない。

| 年 | 入力ファイル | East kyokus | production candidates | reference candidates | 全difference class | 結果 |
|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 100 | 557 | 374 | 374 | 0 | PASS |
| 2011 | 100 | 589 | 395 | 395 | 0 | PASS |
| 2012 | 100 | 573 | 381 | 381 | 0 | PASS |
| 2013 | 100 | 566 | 397 | 397 | 0 | PASS |
| 2014 | 100 | 578 | 397 | 397 | 0 | PASS |
| 2015 | 100 | 577 | 419 | 419 | 0 | PASS |
| 2016 | 100 | 554 | 425 | 425 | 0 | PASS |
| 2017 | 100 | 591 | 426 | 426 | 0 | PASS |
| 2018 | 100 | 554 | 392 | 392 | 0 | PASS |
| 2019 | 100 | 574 | 440 | 440 | 0 | PASS |
| 2020 | 100 | 595 | 441 | 441 | 0 | PASS |
| 2021 | 100 | 553 | 406 | 406 | 0 | PASS |
| 2022 | 100 | 546 | 411 | 411 | 0 | PASS |
| 2023 | 100 | 583 | 465 | 465 | 0 | PASS |
| 2024 | 100 | 570 | 454 | 454 | 0 | PASS |
| **合計** | **1,500** | **8,560** | **6,223** | **6,223** | **0** | **PASS** |

### 2023年全件

共有仕様バグを発見した年度を全件比較した。問題の実牌譜
`2023/2023103019gm-00a9-0000-4b80c963.mjson`もこの範囲に含む。

- `scanned_files`: 168,777
- `production_target_games`: 168,777
- `reference_target_games`: 168,777
- `production_east_kyokus`: 971,994
- `reference_east_kyokus`: 971,994
- `production_candidates`: 756,643
- `reference_candidates`: 756,643
- 全difference class: 0
- `result`: PASS
- `ExitCode`: 0
- `Elapsed`: 02:10:31.4783870

### 2025年全件

- `scanned_files`: 178,888
- `production_target_games`: 178,887
- `reference_target_games`: 178,887
- `production_east_kyokus`: 1,028,072
- `reference_east_kyokus`: 1,028,072
- `production_candidates`: 794,774
- `reference_candidates`: 794,774
- 全difference class: 0
- `result`: PASS
- `ExitCode`: 0
- `Elapsed`: 02:26:00.7766042

件数はpre-fixの2025年全件比較と同一だった。

### 重複を除くpost-fix正式検証範囲

2010〜2024各年100ファイルのうち、2023年の100ファイルは2023年全件に含まれるため
二重加算しない。2025年の監査済み8ファイルと決定論的100/1000ファイルも2025年
全件に含まれるため、別途加算しない。

- files: `6,897 + 1,400 + 168,777 + 178,888 = 355,962`
- target games: `6,897 + 1,400 + 168,777 + 178,887 = 355,961`
- East kyokus: `39,778 + (8,560 - 583) + 971,994 + 1,028,072 = 2,047,821`
- production candidates:
  `28,298 + (6,223 - 465) + 756,643 + 794,774 = 1,585,473`
- reference candidates: 1,585,473
- 全difference class: 0

指定したpost-fix検証範囲ではproduction/referenceが完全一致し、差分は検出され
なかった。これは実装の信頼性を高める結果だが、2009〜2025全年全件を比較した
ものではなく、正しさや未検証入力の不存在を証明するものでもない。2010〜2022年と
2024年は各年100ファイルのみであり、各年全件は未検証である。

## 4. Canonical full exportと最終integrity verification

### Dataset identity

| 項目 | 値 |
|---|---|
| output root | `data/processed/riichi-waits-v1` |
| `dataset_name` | `riichi-waits-v1` |
| `schema_version` | 1 |
| mode | `full` |
| years | 2009〜2025（17年度） |
| scope | `rule_code=00a9`, `aka_flag=true`, `bakaze=E` |
| source | `NikkeTryHard/tenhou-to-mjai`, release `v2.0.0` |
| generator commit | `6904f8661bd414cce081ac524e031e38821a0284` |
| fifth-tile-wait fix | `e596560 Handle Tenhou fifth-tile waits` |

generator metadataはPython 3.12.10、zlib 1.3.1、`worktree_clean=true`だった。

### Full export result

- `scanned_files`: 2,500,236
- `target_games`: 2,500,235
- `east_kyokus`: 14,434,809
- `established_riichis`: 10,706,714
- `output_records`: 10,706,714
- `established_riichis == output_records`: true
- `ExitCode`: 0
- `Elapsed`: 19:55:14.3386946

### Completion markerとserialization

- `manifest.json`: 存在
- `manifest.json.part`: 不存在
- その他の`*.part`: 0件

したがって、checkpoint途中状態ではなくcomplete状態である。serializationはJSON
Lines、UTF-8、gzip compression level 6、gzip `mtime=0`、gzip filename `""`、
`sort_keys=true`、`allow_nan=false`、LFで行った。

### Annual artifact integrity

manifestに記録された2009〜2025の全17年度gzipについて、実ファイルからread-onlyで
`compressed_size_bytes`とSHA256を再計算し、manifest値と比較した。

- PASS: all 17 annual files match manifest size and SHA256
- missing: 0
- size mismatch: 0
- SHA256 mismatch: 0

2023年は168,777ファイル、971,994東場、756,643出力レコードで完走した。共有仕様
バグを発見した`2023/2023103019gm-00a9-0000-4b80c963.mjson`を含み、同年度のpost-fix
reference全件比較もPASSしている。

2025年は178,888ファイル、178,887対象対局、1,028,072東場、794,774出力レコードで
完走し、post-fix reference全件比較もPASSしている。

### Canonical確定とreference validationの範囲

full export正常終了、completion marker成立、checkpoint残骸なし、manifest totals整合、
全17年度artifactの存在・size・SHA256一致、generatorのclean worktree、およびpost-fix
reference検証済み範囲で全difference class 0を確認した。この結果に基づき、
`data/processed/riichi-waits-v1`をcanonical datasetとして確定する。

reference検証の重複除外範囲は、355,962ファイル、355,961対象対局、2,047,821東場、
1,585,473成立リーチである。これは2009〜2025全年全件のreference comparisonでは
ない。canonical full exportの完走とartifact integrityは、未比較年度の全候補が
reference実装と一致することや、実装の絶対的な正しさを証明するものではない。
