# 成立リーチの待ち牌・待ち形・巡目抽出仕様

## 目的

成立したリーチ1件を1レコードとして、リーチ宣言時の手牌、待ち牌、
待ち形、本人の打牌回数を再利用可能な中間データとして抽出する。

このデータは、後続Issueで次の関係を分析するための正解ラベルとなる。

```text
河から観測できる特徴
    -> 実際のリーチ時手牌から求めた待ち牌・待ち形
```

このIssueでは「良形」を定義しない。対子落とし、カンチャン落とし、
筋の安全度、およびそれらを用いた比較集計も対象外とする。

## 対象データ

- データセット: `NikkeTryHard/tenhou-to-mjai` `v2.0.0`
- 対象期間および主分析期間: 2009年から2025年
- 四人打ち鳳凰卓半荘戦
- ファイル名の `rule_code == "00a9"`
- 最初の `start_game` の `aka_flag is True`
- `start_kyoku.bakaze == "E"` の東場のみ

production経路の対象対局判定、MJAI読み込み、局分割、東場抽出は、既存の
`load_mjai()`、`is_target_game()`、`split_kyoku()`、
`filter_east_kyokus()` を再利用する。対象期間の既定値を
2020年から2025年に狭めてはならない。

## 処理レイヤー

責務を次の3層に分離する。

1. 牌種と麻雀ルール
   - 赤牌の正規化
   - 完成形の列挙
   - 待ち牌と待ち形の判定
   - MJAIイベントや保存形式には依存しない
2. MJAI状態再生と成立リーチ抽出
   - 初期手牌、ツモ、打牌、副露、暗槓をイベント順に反映する
   - 成立リーチと宣言牌の打牌番号を求める
3. レコード生成と保存
   - 対象対局・東場フィルタを適用する
   - 1成立リーチを1レコードに変換する
   - 保存形式固有処理を分析ロジックから分離する

## 成立リーチの定義

次の3イベントが局内で連続し、全て同一actorを指す場合だけを
成立リーチとする。

1. `reach`
2. 同一actorのリーチ宣言牌 `dahai`
3. 同一actorの `reach_accepted`

`reach` と `dahai` の間、または `dahai` と `reach_accepted` の間に、
別のイベントが入ってはならない。

次の場合は、妥当な未成立リーチとしてレコードを生成しない。

- `reach`、同一actorの `dahai`、`hora` が隣接し、宣言牌にロンが
  発生した場合。このとき `hora.target` はreach actorと一致し、かつ
  `hora.actor != hora.target` でなければならない。`hora` に `pai` keyが
  存在する場合は、その値が文字列の有効牌であり、宣言牌 `dahai.pai` と
  正規化後の同一牌種でなければならない。`pai` key自体が存在しない場合は、
  直前の宣言牌 `dahai.pai` を和了牌として扱う。keyが存在して値がnull、
  文字列以外、または不正牌の場合は省略扱いせず拒否する。自己和了形式や
  無関係な `hora` も正常な宣言牌ロンとせず、不正イベント列として拒否する
- `reach`、同一actorの `dahai`、`ryukyoku` の順で、
  `reach_accepted` に到達しなかった場合

これ以外の欠落、actor不一致、予期しないイベント順は不正MJAIとして
扱う。production経路では、既存の `is_dealer_double_riichi()` と同じ成立条件を
共有し、親の第一打リーチに限らない全actor向けの判定を別実装として
重複させない。

## 判定時点の手牌

待ち牌・待ち形は、次の時点の手牌から求める。

```text
reach
同一actorのリーチ宣言牌dahai
---------- 待ち判定スナップショット ----------
reach_accepted
```

すなわち、リーチ宣言牌を手牌から取り除いた直後、
`reach_accepted` を処理する前の13枚相当の手牌である。
`reach_accepted` は手牌を変化させるイベントとして使わず、
リーチが成立したことの確認にのみ使用する。

暗槓がない場合、concealed handは物理的に13枚である。暗槓が `k` 個
ある場合、concealed handの物理枚数は `13 - 3k` 枚となる。固定済みの
暗槓を1面子3枚相当として数えた論理手牌は13枚相当であり、待ち判定は
concealed handと固定面子を合わせて行う。

手牌スナップショットは宣言牌 `dahai` の適用後に作成する。
先に `reach` 時点の14枚相当手牌で待ちを求めてはならない。

## riichi_discard_number

`riichi_discard_number` は、リーチ宣言牌がそのactorの第何打かを表す
1始まりの整数とする。

```text
riichi_discard_number
    = 宣言牌dahaiより前の同一actorのdahai数 + 1
```

卓全体の巡目、ツモ回数、他家の打牌数は使用しない。他家の鳴きで
ツモ順が変わっていても、本人の `dahai` だけを数える。本人が副露後に
リーチすることはできないため、成立リーチ者に過去のチー、ポン、
大明槓または加槓が見つかった場合は不正MJAIとする。暗槓は許容する。

## 1レコードの単位と識別子

成立リーチ1件につき1レコードを生成する。同じ局で複数人が
リーチした場合は、actorごとに別レコードとする。

最低限、次の識別情報を持つ。

- `riichi_id`
- `year`
- `filename`
- `start_kyoku_line`
- `reach_line`
- `dahai_line`
- `reach_accepted_line`
- `bakaze`
- `kyoku`
- `honba`
- `kyotaku`
- `oya`
- `actor`
- `is_dealer`

行番号は展開後のMJAI JSON Linesを1始まりで数えた物理行番号とする。
`filename + start_kyoku_line` で局を、`filename + reach_line` で成立リーチを
一意に識別する。`riichi_id` は後者から決定論的に生成し、後続の
特徴量データと結合できる安定キーとする。

## リーチ情報

各レコードは次を持つ。

- `riichi_discard_number`
- `riichi_declaration_tile`
  - MJAIイベントの生牌表記
  - 赤牌なら `5mr`、`5pr`、`5sr` を維持する
- `riichi_declaration_tile_kind`
  - 赤を通常5へ正規化した牌種

## 手牌状態の再生

`start_kyoku.tehais[actor]` を初期concealed handとし、
リーチ宣言牌までのイベントを順番に適用する。

この層は完全な麻雀行動合法性検証器ではない。牌所有、副露状態、
打牌後枚数など、正確な手牌再現に必要な整合性を検証する。

- `tsumo`
  - 同一actorの `pai` をconcealed handへ1枚追加する
- `dahai`
  - 同一actorの `pai` と同じ物理牌を1枚除く
  - `5m` と `5mr` は状態更新時には別の物理牌として扱う
  - 適用直後の全actorについて、concealed handの物理枚数が
    `13 - 3 * fixed_meld_count` であることを検証する
- `chi` / `pon` / `daiminkan`
  - actorのhandから `consumed` を除き、固定面子を追加する
  - 四人麻雀の `chi` は `actor == (target + 1) % 4` を満たすことを
    検証する
- `ankan`
  - actorのhandから `consumed` 4枚を除き、閉じた固定面子を追加する
- `kakan`
  - actorのhandから追加牌 `pai` を1枚除き、既存ポンを槓へ更新する
- 他actorの手牌イベント
  - リーチ者のconcealed handは変更しない

存在しない物理牌のツモ切り・手出し、5枚目の同一牌種、不正な
`consumed`、固定面子数とconcealed hand枚数の不整合はエラーとする。

成立リーチの宣言牌打牌直後のconcealed handと固定面子は、
この抽出境界で `calculate_hand_waits()` に渡して検証する。
`wait_tiles` が空なら不正MJAIとして例外にし、`EstablishedRiichi` を
生成しない。待ち探索アルゴリズム自体は手牌再生層へ重複実装しない。

## actor_discards_before_riichi

後続Issueで河から見える対子落とし・カンチャン落としを判定できるよう、
同一actorのリーチ宣言牌までの全 `dahai` を順番に保持する。
名称中の `before_riichi` は `reach_accepted` より前という意味とし、
リーチ宣言牌自身を含む。

各要素は少なくとも次を保持する。

- `discard_number`
- `tile`
  - MJAIの生牌表記
- `tile_kind`
  - 赤を通常5へ正規化した牌種
- `tsumogiri`
- `event_index`
  - `start_kyoku` を0とするkyoku内の `dahai` イベント位置
- `is_riichi_declaration`
- `was_called`
- `call_type`
  - 呼ばれていない場合は `null`
  - 呼ばれた場合は `chi`、`pon`、`daiminkan` のいずれか
- `called_by_actor`
  - 呼ばれていない場合は `null`
- `call_event_index`
  - kyoku内のcallイベント位置
  - 呼ばれていない場合は `null`

`was_called` は `call_type is not null` から再計算可能な派生値とする。
ロンは鳴きに含めない。リーチ宣言牌は `reach_accepted` の直後に他家から
鳴かれることがあるため、待ち判定スナップショットを保存した後も
局イベントを確認し、その宣言牌が最終的に鳴かれたかを再現できるように
する。

`ActorDiscard` における位置情報の正本はkyoku内の `event_index` と
`call_event_index` とする。いずれも `start_kyoku` を0とする0始まりの
イベント位置である。`dahai_line`、`call_event_line`、`reach_line`、
`declaration_dahai_line`、`reach_accepted_line` 等の物理行番号は、
source-aware reader、comparison adapter、audit CLIが元ファイル追跡用metadata
として付与する。production/referenceの意味的な河比較にはkyoku内indexを使い、
物理行番号はcandidate key、差分診断、元MJAI追跡に使う。同じ位置情報を二つの
正本として扱わない。

このIssueでは、上記の河イベントから対子落とし等の特徴量を計算しない。

## 牌種と赤5の正規化

待ち判定用の牌種は次の34種とし、この順序を正規順序とする。

```text
1m ... 9m, 1p ... 9p, 1s ... 9s, E, S, W, N, P, F, C
```

赤5は次のように正規化する。

```text
5mr -> 5m
5pr -> 5p
5sr -> 5s
```

赤5と通常5は、手牌状態を再生するときは異なる物理牌として保持し、
完成形と待ち形を判定するときは同じ牌種として扱う。

待ち判定層では赤5を通常5へ正規化して扱う。`5mr`、`5pr`、`5sr` が
物理的に複数存在する等の不正MJAIの検査はこの層では行わず、後続の
MJAI手牌再生層で生牌を保持した状態で行う。

現段階のMJAI手牌再生層では、各actorのconcealed handと固定面子を
合わせた手牌状態内で同じ赤5が複数存在する場合を拒否する。局全体での
`5mr`、`5pr`、`5sr` の物理的一意性を追跡するグローバル牌台帳は、
完全なMJAI合法性検証の責務とし、Issue #24の必須範囲には含めない。

- `wait_tiles` に `5mr`、`5pr`、`5sr` は出力しない
- 通常5と赤5を別の待ち種類として数えない
- 5待ちは正規化後の `5m`、`5p`、`5s` 1種類とする
- 同一actorが所有する通常牌と赤牌の合計が4枚なら、その牌種を
  新たな和了牌候補にしない
- 宣言牌と河履歴には生牌と正規化牌の両方を保存する

## 待ち情報の正本

`wait_tiles` と `wait_details` を待ち情報の正本とする。

### wait_tiles

リーチ宣言牌を除いた直後の手牌に1枚加えることで和了形となる、
異なる正規化牌種の集合である。重複を持たず、34種の正規順序で保存する。

河、副露、ドラ表示牌等に何枚見えているか、山に実際に残っているか、
フリテンかどうかは `wait_tiles` に影響させない。これは残り枚数ではなく、
手牌構造上の和了牌種である。ただし同一actorが既に4枚全てを所有している
牌種は、5枚目を加えられないため候補外とする。

### wait_details

`wait_details` は次の3項関係を保持するレコードの配列とする。

```text
wait_tile × hand_type × wait_shape
```

各要素は次の形を持つ。

```json
{
  "wait_tile": "5p",
  "hand_type": "standard",
  "wait_shape": "kanchan"
}
```

同じ待ち牌に異なる待ち形または異なるhand typeの解釈があれば、
その全てを別要素として保持する。完成形の最初の1分解だけを採用しては
ならない。同一の `wait_tile × hand_type × wait_shape` が複数の完成分解から
得られた場合、その3項関係は1件に重複排除する。完成分解の個数自体は
保存対象としない。

`wait_tiles` と `wait_details` は次の不変条件を満たさなければならない。

```text
set(wait_tiles) == {detail.wait_tile for detail in wait_details}
```

`wait_details` は `wait_tile`、`hand_type`、`wait_shape` の定義順で
決定論的に並べる。

## hand_type

`hand_type` は次のいずれかとする。

- `standard`
  - 固定面子を含めて4面子1雀頭となる形
- `chiitoitsu`
  - 異なる7牌種の7対子
  - 同一牌4枚を2対子として数えない
  - 固定面子がある場合は成立しない
- `kokushi`
  - 国士無双
  - 固定面子がある場合は成立しない

同じ和了牌で標準形と七対子等の両方が成立する場合は、両方の
`wait_details` を保持する。

## wait_shape

`wait_shape` は次のいずれかとする。

- `ryanmen`
  - 数牌順子の端を完成する
  - `23` の `1`・`4`、`78` の `6`・`9` は両面に含む
- `kanchan`
  - 数牌順子の中央を完成する
- `penchan`
  - `12` の `3`、または `89` の `7` を完成する
- `shanpon`
  - 対子を刻子へ完成する
- `tanki`
  - 単独牌を雀頭へ完成する
  - 七対子の最後の対子もこの値を使い、`hand_type` で区別する
- `kokushi_single`
  - 既に1対子があり、不足している1種だけを待つ国士無双
- `kokushi_13men`
  - 13種を1枚ずつ持つ国士無双13面待ち

`hand_type` と `wait_shape` の有効な組合せは次のとおりとし、これ以外は
不正値として拒否する。

- `standard`: `ryanmen`、`kanchan`、`penchan`、`shanpon`、`tanki`
- `chiitoitsu`: `tanki`
- `kokushi`: `kokushi_single`、`kokushi_13men`

標準形では、和了牌を完成形の各分解に割り当てた全ての解釈を調べる。
同一牌を順子側に割り当てる場合と雀頭側に割り当てる場合の両方が
成立するなら、両方を `wait_details` に残す。

「複合形」を上記とは別の単一分類へ押し込めない。複合性は、同一待ち牌の
複数detail、複数の `wait_shape`、および複数の `wait_tile` として保持する。

## 派生値

次の値は正本ではなく、`wait_tiles` と `wait_details` から再計算可能な
派生値とする。保存する場合も、出力前に正本との一致を検証する。

### wait_tile_count

```text
wait_tile_count = len(wait_tiles)
```

残り枚数ではなく、異なる正規化和了牌種の数である。

### wait_shapes

```text
wait_shapes = distinct(detail.wait_shape for detail in wait_details)
```

待ち形の定義順で決定論的に保存する。

### contains_ryanmen

```text
contains_ryanmen =
    any(detail.wait_shape == "ryanmen" for detail in wait_details)
```

多面張や複合形でも、両面解釈を1つ以上含めば `True` とする。

### is_pure_ryanmen

次を全て満たす場合だけ `True` とする。

```text
wait_tile_count == 2
全wait_detailsのhand_typeがstandard
全wait_detailsのwait_shapeがryanmen
```

したがって、3種以上の両面由来待ちは `contains_ryanmen=True` だが
`is_pure_ryanmen=False` となる。同じ待ち牌に単騎等の別解釈があれば
`False` とする。

### is_multiwait

```text
is_multiwait = wait_tile_count >= 3
```

- 通常の2種両面: `False`
- 2種シャンポン: `False`
- 2種の複合待ち: `False`
- 3種以上の待ち: `True`
- 国士無双13面待ち: `True`

2種でも複雑な形を失わないため、`is_multiwait` だけで待ちを分類せず、
必ず正本の `wait_tiles` と `wait_details` を利用できるようにする。

## 「良形」の非定義

このIssueでは、次のいずれについても「良形」かどうかを決めない。

- 純粋な両面
- 両面を含む多面張
- 両面を含まない多面張
- シャンポン
- 単騎を含む複合形

後続分析は正本と派生値を組み合わせ、分析目的ごとに「良形」を定義する。
基盤データに `is_good_wait`、`good_shape` 等の列を設けない。

## 赤5、暗槓、特殊形、複合形の具体例

### 赤5

`4s 5sr` を含む両面ターツは、正規化後の `4s 5s` として判定し、
待ちは `3s, 6s` とする。`3sr` や `6sr` は有効な牌表記ではない。

5そのものが待ちの場合、通常5と赤5のどちらで和了し得るかを
`wait_tile_count` に別々に数えず、正規化牌種 `5m`、`5p` または `5s`
1種類だけを保存する。

### 暗槓

暗槓 `9m 9m 9m 9m` が固定済みで、concealed handが
`123p 789p EE 45s` の10枚なら、固定暗槓を1面子として待ちは
`3s, 6s` となる。両方のdetailは `standard / ryanmen` である。

暗槓牌をactorの所有牌数に含めるため、この例で `9m` を5枚目の
和了牌候補にしてはならない。

### 七対子

`11m 22m 33p 44p 55s 66s E` は `E` 待ちである。

```text
E × chiitoitsu × tanki
```

標準形としても成立する別解釈がある牌姿では、七対子と標準形のdetailを
両方保存する。

### 国士無双

`1m 9m 1p 9p 1s 9s EE S W N P F` は `C` だけが不足しており、
次のdetailを持つ。

```text
C × kokushi × kokushi_single
```

`1m 9m 1p 9p 1s 9s E S W N P F C` は13種全てが待ちであり、
各牌について `kokushi / kokushi_13men` を持つ。

### 多面張

`34567m 123p 789p EE` は `2m, 5m, 8m` 待ちで、全detailが
`standard / ryanmen` である。

```text
contains_ryanmen = True
is_pure_ryanmen = False
is_multiwait = True
```

`2345678m 123p 789p` は `2m, 5m, 8m` 待ちで、全detailが
`standard / tanki` である。

```text
contains_ryanmen = False
is_pure_ryanmen = False
is_multiwait = True
```

### 同一牌の複数解釈

`123m 456m 789m 4556p` は `5p` だけが待ちである。`5p` を加えた
`45556p` は `456p + 55p` となるが、追加した5をどちらに割り当てるかで
次の2解釈が成立する。

```text
5p × standard × kanchan
5p × standard × tanki
```

この場合、`wait_tile_count=1`、`is_multiwait=False` である。
`wait_shapes` をどちらか一方に限定してはならない。

## 人工手牌の必須テストケース

牌姿はリーチ宣言牌を除いた直後のものとする。`123m` は
`1m, 2m, 3m`、`EE` は `E, E` を表す。固定面子欄が空なら暗槓なしで、
concealed handは物理的に13枚である。

`wait_details` の表記は `wait_tile / hand_type / wait_shape` とする。
人工ケースはR1からR16までの16件とする。

| ID | Concealed hand | 固定面子 | `wait_tiles` | `wait_details` |
|---|---|---|---|---|
| R1 純両面 | `123m 123p 789p EE 45s` | なし | `3s, 6s` | `3s/standard/ryanmen`, `6s/standard/ryanmen` |
| R2 カンチャン | `123m 123p 789p EE 46s` | なし | `5s` | `5s/standard/kanchan` |
| R3 ペンチャン | `123m 456m 789p EE 12s` | なし | `3s` | `3s/standard/penchan` |
| R4 シャンポン | `123m 456m 789m 55p 77s` | なし | `5p, 7s` | `5p/standard/shanpon`, `7s/standard/shanpon` |
| R5 単騎 | `123m 456m 789m 123p 5s` | なし | `5s` | `5s/standard/tanki` |
| R6 両面由来3面張 | `34567m 123p 789p EE` | なし | `2m, 5m, 8m` | `2m/standard/ryanmen`, `5m/standard/ryanmen`, `8m/standard/ryanmen` |
| R7 両面なし3面張 | `2345678m 123p 789p` | なし | `2m, 5m, 8m` | `2m/standard/tanki`, `5m/standard/tanki`, `8m/standard/tanki` |
| R8 両面＋単騎 | `2333456m 123p 789p` | なし | `1m, 2m, 4m, 7m` | `1m/standard/ryanmen`, `2m/standard/tanki`, `4m/standard/ryanmen`, `7m/standard/ryanmen` |
| R9 同一牌の形重複 | `123m 456m 789m 4556p` | なし | `5p` | `5p/standard/kanchan`, `5p/standard/tanki` |
| R10 七対子 | `11m 22m 33p 44p 55s 66s E` | なし | `E` | `E/chiitoitsu/tanki` |
| R11 国士単騎 | `1m 9m 1p 9p 1s 9s EE S W N P F` | なし | `C` | `C/kokushi/kokushi_single` |
| R12 国士13面 | `1m 9m 1p 9p 1s 9s E S W N P F C` | なし | 13種の么九牌全て | 各待ち牌について `kokushi/kokushi_13men` |
| R13 赤5を含む両面 | `123m 123p 789p EE 4s 5sr` | なし | `3s, 6s` | R1と同じ2detail |
| R14 暗槓あり両面 | `123p 789p EE 45s` | `ankan: 9999m` | `3s, 6s` | `3s/standard/ryanmen`, `6s/standard/ryanmen` |
| R15 両面＋シャンポン複合多面張 | `22234567m 22p 789s` | なし | `2m, 5m, 8m, 2p` | `2m/standard/ryanmen`, `2m/standard/shanpon`, `5m/standard/ryanmen`, `8m/standard/ryanmen`, `2p/standard/shanpon` |
| R16 異なる完成分解の全探索 | `11122233m 456p EE` | なし | `3m, E` | `3m/standard/penchan`, `3m/standard/shanpon`, `E/standard/shanpon` |

R12の期待値を省略せずに表すと、次のとおりである。

```text
wait_tiles =
    1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C

wait_details =
    1m/kokushi/kokushi_13men
    9m/kokushi/kokushi_13men
    1p/kokushi/kokushi_13men
    9p/kokushi/kokushi_13men
    1s/kokushi/kokushi_13men
    9s/kokushi/kokushi_13men
    E/kokushi/kokushi_13men
    S/kokushi/kokushi_13men
    W/kokushi/kokushi_13men
    N/kokushi/kokushi_13men
    P/kokushi/kokushi_13men
    F/kokushi/kokushi_13men
    C/kokushi/kokushi_13men
```

R13の期待値は、赤5を通常5へ正規化した上で次のとおりである。

```text
wait_tiles = 3s, 6s
wait_details =
    3s/standard/ryanmen
    6s/standard/ryanmen
```

R6の`5m`では、同一完成分解の`345m`側または`567m`側へ和了牌を
割り当てられ、どちらも `5m/standard/ryanmen` となる。同様にR8の`4m`も、
同一完成分解の`234m`側または`456m`側へ和了牌を割り当てられ、どちらも
`4m/standard/ryanmen` となる。R6とR8は、主に同一完成分解内で和了牌を
複数の構成要素へ割り当てるケースを検査する。

これらの割り当ては全て調べるが、同一の
`wait_tile / hand_type / wait_shape` に到達した重複数は保存しない。
したがって、R6の`5m`とR8の`4m`は、それぞれ `wait_details` 上では
1件へ重複排除する。

R15は、同一待ち牌`2m`が両面とシャンポンの両方として解釈できる
複合多面張の境界ケースである。期待値は次のとおりである。

```text
wait_tiles = 2m, 5m, 8m, 2p
wait_details =
    2m/standard/ryanmen
    2m/standard/shanpon
    5m/standard/ryanmen
    8m/standard/ryanmen
    2p/standard/shanpon

contains_ryanmen = True
is_pure_ryanmen = False
is_multiwait = True
```

R16の`3m`では、次の異なる標準形の完成分解が成立する。

```text
111m + 222m + 333m + 456p + EE
123m + 123m + 123m + 456p + EE
```

前者では追加した`3m`を刻子へ割り当てる `shanpon`、後者では順子へ
割り当てる `penchan` となる。R16は、最初に見つかった完成分解で
打ち切らず、異なる完成分解を最後まで探索することを直接検査する。

各ケースで `wait_tiles` と `wait_details` の完全一致を検査する。
部分集合の一致、待ち牌数だけの一致、`contains_ryanmen` だけの一致では
テスト成功としない。

派生値について、少なくとも次を同時に検査する。

| ID | `contains_ryanmen` | `is_pure_ryanmen` | `is_multiwait` |
|---|---:|---:|---:|
| R1 | `True` | `True` | `False` |
| R2-R5 | `False` | `False` | `False` |
| R6 | `True` | `False` | `True` |
| R7 | `False` | `False` | `True` |
| R8 | `True` | `False` | `True` |
| R9-R11 | `False` | `False` | `False` |
| R12 | `False` | `False` | `True` |
| R13-R14 | `True` | `True` | `False` |
| R15 | `True` | `False` | `True` |
| R16 | `False` | `False` | `False` |

## 人工MJAIイベントの必須テストケース

手牌単体テストとは別に、少なくとも次をMJAIイベント列から検証する。

- 親の成立リーチ
- 子の成立リーチ
- 第一打リーチ
- 7枚目の本人打牌によるリーチ
- 遅巡リーチ
- 1局中の複数成立リーチ
- 他家のチー、ポン、暗槓、大明槓が存在する局
- リーチ者自身の暗槓後リーチ
- 赤5のツモ、手出し、ツモ切り、宣言牌
- `reach` はあるが宣言牌で放銃し、`reach_accepted` がないケース
  - `hora` が宣言牌 `dahai` の直後で、`hora.target` と宣言actorが一致し、
    `hora.actor != hora.target` であることを検査する
  - `hora.pai` が存在すれば正規化後の宣言牌との一致を検査し、key自体が
    なければ直前の宣言牌を和了牌として扱う。keyが存在するnull、不正型、
    不正牌は拒否する
- `reach` はあるが `ryukyoku` となり、`reach_accepted` がないケース
- `reach` と `dahai` のactor不一致
- `reach_accepted` のactor不一致
- 3イベントの間に予期しないイベントが入るケース
- 孤立した `reach_accepted`
- 手牌に存在しない物理牌の `dahai`
- 全actorの打牌直後枚数が `13 - 3 * fixed_meld_count` と異なるケース
- 不正な方向からの `chi`
- 副露済みactorの不正な成立リーチ
- 待ちが1種類も得られない不正な成立リーチ

イベントテストでは、待ち判定スナップショットが宣言牌 `dahai` 適用後、
`reach_accepted` 適用前であることを明示的に検査する。

## 不正・不完全MJAIの扱い

対象ファイルを黙ってskipせず、ファイル単位で処理を失敗させる。
例外には可能な限り `year`、`filename`、物理行番号、actorを含める。
途中まで生成したレコードを正常な年次出力として確定してはならない。

次を不正または不完全とする。

- JSONとして読めない行、JSON objectでないイベント
- 不正なファイル名
- 空ファイル、先頭が `start_game` でないファイル
- `aka_flag` の欠落またはbool以外
- 対象局の `start_kyoku.tehais` 欠落・不正
- `start_kyoku` / `end_kyoku` の不整合
- actorが0から3の範囲外
- 不正牌表記または関連手牌中の不明牌 `?`
- 手牌に存在しない生牌の削除
- 正規化後に同一牌種が5枚以上となる状態
- 不正な副露・槓の `consumed`
- 成立リーチ3イベントの欠落、actor不一致、順序不正
- 対応する `reach` のない `reach_accepted`
- 成立リーチ者が既に開いた固定面子を持つ
- 宣言牌打牌後の論理手牌枚数が13枚相当でない
- 成立確認後に `wait_tiles` が空となる
- `wait_tiles` と `wait_details` の不変条件違反
- 正本から再計算した派生値と保存値の不一致

非対象の `rule_code`、`aka_flag is False`、南場・西場はエラーではなく
対象外である。宣言牌での放銃等により `reach_accepted` に到達しない
正規のイベント列もエラーではなく、未成立として除外する。

## 独立検証

本番実装とは独立したreference経路を設け、成立リーチの検出、手牌再生、
待ち判定をproduction/reference間で比較する。referenceは件数だけでなく、
成立リーチ1件ごとの状態と待ち情報を独立に生成しなければならない。

既存MJAI処理の再利用と成立リーチ判定の重複禁止はproduction経路に適用する。
独立reference検証は、productionと同じ不具合を共有していないことを検証するための
意図的な例外である。reference側では、本番実装と独立に次を再実装する。

- 対象対局判定
- MJAI読込と状態再生
- 成立リーチ検出
- 待ち判定
- 派生値計算

この例外は検証用reference経路だけに適用し、production経路の共通化方針は
変更しない。

### reference実装の依存境界

reference packageから次のproduction主要モジュールをimportしてはならない。

- `mahjong_analysis.riichi`
- `mahjong_analysis.hand_waits`
- `mahjong_analysis.riichi_wait_records`
- `mahjong_analysis.tiles`
- `mahjong_analysis.mjai`

上記モジュールのprivate helperをコピーして名前だけ変更することも禁止する。
productionとreferenceの両方をimportする処理は比較adapterに限定し、reference
package内へ置かない。

reference側は、次の要素をproductionから独立して定義・実装する。

- 34牌種の順序、生牌表記、赤5の正規化表
- reference専用の手牌、固定面子、捨て牌、待ち詳細、候補レコードの型
- MJAI JSON Linesの読込、gzip判定、物理行番号の付与
- 対象対局の `00a9`、`aka_flag is True`、東場判定
- 成立リーチ検出、手牌再生、待ち判定、派生値計算

referenceの状態表現はproductionの状態クラスを使わない。concealed handは
生牌別の多重集合等、productionと異なる構造で保持する。禁止importは
reference package全体のAST検査で機械的に確認する。また、production主要
関数を呼び出せない状態でもreference単体テストが通ることを確認する。

### 独立待ち判定

referenceの標準形判定は、productionのように34牌種を1枚ずつ加えた14枚の
完成形を全分解し、和了牌の所属先を調べる方式を使用しない。13枚の手牌から
未完成部分を先に予約し、残余が完成面子だけで構成可能かを調べる。

fixed meld数を `f`、concealed hand内に必要な完成面子数を `q = 4 - f`
とする。成立リーチのfixed meldには `ankan` だけを許容する。

単騎は、concealed hand内の各牌を1枚ずつ未完成雀頭として予約する。残りが
`q` 個の完成面子で構成できる場合、予約牌を `standard/tanki` とする。

面子待ちは、雀頭候補2枚と未完成面子の2枚を先に予約する。雀頭と未完成
面子の全候補を検討し、残りが `q - 1` 個の完成面子で構成できる場合に、
未完成面子から次の待ちを直接列挙する。

- 同一牌2枚: 同じ牌の `standard/shanpon`
- 同一スートの `1-2`: `3` の `standard/penchan`
- 同一スートの `8-9`: `7` の `standard/penchan`
- 上記以外の同一スートの隣接2牌: 両端の `standard/ryanmen`
- 同一スートの1牌飛ばし: 中間牌の `standard/kanchan`

完成面子部分の構成可能性は、数牌と字牌を次のように判定する。

- 数牌は、各スートの9要素枚数ベクトルを、順子と刻子から事前生成した
  完成面子構成可能集合と照合する。照合結果として、そのスートを構成する
  完成面子数も求める。
- 字牌は、残余の各牌種の枚数が0または3でなければならない。3枚ある字牌
  1種類につき完成刻子1面子として数える。
- 数牌側で成立した完成面子数と字牌刻子数の合計は、要求された完成面子数と
  一致しなければならない。

単騎では、1牌を未完成雀頭として予約した残余が `q` 個の完成面子になることを
要求する。雀頭と面子待ち断片を予約する場合は、雀頭2枚と2枚断片を除いた残余が
`q - 1` 個の完成面子になることを要求する。

例えば `EEE 123m 456p NN 45s` では、`NN` を雀頭、`45s` を未完成面子、
`EEE` を完成字牌刻子として数える。残りの `123m`、`456p` と合わせて3面子が
成立するため、`3s/standard/ryanmen` と `6s/standard/ryanmen` を検出する。

残余と雀頭が確定した後は構成可能性の真偽だけを使用し、完成分解自体は列挙
しない。雀頭と未完成面子の探索を途中で打ち切ってはならない。同じ
`tile / hand_type / wait_shape` へ至る経路だけを1件へ重複排除する。

七対子と国士無双は標準形と別に直接判定し、標準形を含む他のhand typeと
重なる結果も失わず統合する。

- 七対子はfixed meldなしで、6種類が各2枚、1種類が1枚の場合だけ、その
  単牌を `chiitoitsu/tanki` とする。四枚使いを2対子と数えない。
- 国士13面は13種の幺九牌が各1枚の場合とし、全13種を
  `kokushi/kokushi_13men` とする。
- 国士1種待ちは12種の幺九牌と、そのうち1種の対子から成る場合とし、
  不足牌を `kokushi/kokushi_single` とする。

待ち候補は、referenceが独自に数えたconcealed handとfixed meldの所有枚数が
4枚未満の場合だけ採用する。赤5は生牌状態では通常5と区別し、待ち牌と
待ち詳細では通常5へ正規化する。`wait_tiles`、`wait_details`、`wait_shapes`
は仕様順へ決定論的に整列し、派生値もreference側で独立に計算する。

### 成立リーチ検出と手牌再生

referenceは `start_kyoku.tehais` から局イベントを独自に再生する。
`tsumo`、`dahai`、`chi`、`pon`、`daiminkan`、`ankan`、`kakan` を扱い、
`dahai` と `consumed` は指定された生牌そのものを手牌から除く。
`chi`、`pon`、`daiminkan` の `pai` は他家の捨て牌、`consumed` はactorが
手牌から出す牌として扱う。`kakan` は追加牌を除き、対応する既存ponを
更新する。`dora` は状態非変更イベントとする。

reference側でも、牌所有、副露状態、打牌後枚数、同一正規化牌種の5枚所有、
actor内の同種赤5重複、chiの方向、kakan対象ponの存在を検査する。完全な
麻雀行動合法性検証器にはしないが、不正・不完全MJAIを黙ってskipしない。

成立リーチは、隣接する次の3イベントをreference側で独自に検出する。

1. `reach`
2. 同一actorの宣言牌 `dahai`
3. 同一actorの `reach_accepted`

宣言牌 `dahai` を手牌へ適用した直後に、次を不変スナップショットとして固定
する。

- `concealed_tiles_after_discard`
- `fixed_melds`
- `riichi_discard_number`
- 本人第1打からリーチ宣言打牌までという河履歴の対象範囲

河履歴の対象範囲は宣言牌 `dahai` 後に増やさない。一方、その範囲に含まれる
各捨て牌の `was_called`、`call_type`、`called_by_actor`、`call_event_index` は、
宣言牌 `dahai` 後も局終了までイベントを追跡し、後から鳴かれた場合に更新する。

特に、次のようにリーチ成立後に宣言牌が他家から鳴かれる場合も、最終的な
`actor_discards_before_riichi` へcall metadataを反映する。

```text
reach
dahai
reach_accepted
他家のchi / pon / daiminkan
```

`reach_accepted` はリーチ成立を確定するイベントであり、河のcall metadataの
追跡終了点ではない。reference candidateのconcealed hand、fixed melds、本人の
打牌数は宣言牌 `dahai` 直後の状態を維持し、河履歴は対象範囲を固定したまま
call metadataだけを局終了まで追跡した最終値をproductionと比較する。

`reach -> dahai -> hora` は3イベントが隣接し、`hora.target` がreach actorで、
`hora.actor != hora.target` の場合だけ正常な未成立リーチとする。`hora` に
`pai` keyが存在する場合は、有効な牌文字列であり、宣言牌と同一正規化牌種で
なければならない。`pai` key自体が存在しない場合は、直前の宣言牌を和了牌と
して扱う。keyが存在するnull、不正型、不正牌は省略扱いせず拒否する。自己和了
形式や無関係な `hora` は不正イベント列として拒否する。
`reach -> dahai -> ryukyoku` も未成立として除外する。それ以外の不正な成立列は、
source path、物理行番号、可能ならactorを含む例外にする。

### candidate key

production/referenceの候補集合を次のキーで比較する。

```text
(relative_source_path, start_kyoku_line, reach_line)
```

`relative_source_path` は比較CLIへ指定したraw rootからの相対パスとし、
区切り文字を `/` へ統一する。行番号は展開後のMJAI JSON Linesを1始まりで
数えた物理行番号とする。

`actor`、宣言牌、宣言 `dahai` 行、`reach_accepted` 行、手牌、待ち情報は
キーへ含めず比較フィールドとする。これらをキーへ含めて誤りを候補集合差へ
分散させてはならない。同一側でcandidate keyが重複した場合は、辞書上書き
せずエラーとする。

### 正式比較項目

candidate key集合に加え、同じkeyの成立リーチについて次を完全比較する。

- `actor`
- 宣言 `dahai` と `reach_accepted` の物理行番号
- `riichi_discard_number`
- `riichi_declaration_tile`
- `riichi_declaration_tile_kind`
- `concealed_tiles_after_discard`
- `fixed_melds`
- `actor_discards_before_riichi`
- `wait_tiles`
- `wait_tile_count`
- `wait_details`
- `wait_shapes`
- `contains_ryanmen`
- `is_pure_ryanmen`
- `is_multiwait`

`actor_discards_before_riichi` は監査用の参考情報ではなく、最初から正式比較
対象とする。各捨て牌について次を比較する。

- `discard_number`
- raw tile (`tile`)
- normalized tile (`tile_kind`)
- `tsumogiri`
- `event_index`
- `is_riichi_declaration`
- `was_called`
- `call_type`
- `called_by_actor`
- `call_event_index`

`event_index` と `call_event_index` は、対象kyoku内の0始まりイベントindexと
する。河履歴の順序は意味を持つため、比較前に並べ替えてはならない。

concealed handは赤牌を維持した生牌の多重集合、fixed meldはmeld typeと
生牌の多重集合として比較し、格納順だけの差は無視する。`wait_tiles` と
`wait_details` も仕様順へ整列して比較する。ただし、未知の牌、重複禁止値、
赤牌と通常牌の違い、同一待ち牌の異なる解釈など、意味のある差を正規化で
隠してはならない。

### 比較結果

比較結果は少なくとも次の分類を分離して出力する。

- `production_only`
  - production側だけに存在するcandidate key
- `reference_only`
  - reference側だけに存在するcandidate key
- `field_mismatch`
  - 同じcandidate keyで比較フィールドが異なる
- `scope_mismatch`
  - 対象対局判定または東場kyoku集合が異なる
- `processing_error`
  - 片側または両側の読込、再生、判定が例外になった

`field_mismatch` は比較フィールド単位で、candidate key、actor、production値、
reference値、関連イベント行を保持する。不一致から元MJAIを追えるよう、
source path、`start_kyoku_line`、`reach_line` と周辺イベントを出力する。
片側の例外や不正MJAIを候補なしへ変換してはならない。出力順はcandidate keyと
比較フィールドの固定順で決定論的にする。

必須reference検証では、production側だけの例外、reference側だけの例外、
両側が異なる理由で発生させた例外を含め、`processing_error` が1件でも残る
段階をPASSとしてはならない。各errorは元MJAIまで追跡し、原因を次のいずれかに
分類して調査する。

- productionの不具合
- referenceの不具合
- 仕様不足
- 元牌譜データの異常

現段階では `processing_error` のallowlistを設けない。元牌譜データの異常が
実際に見つかり、allowlistが必要になった場合は、その時点で本仕様を明示的に
変更する。黙ってskipしたり、例外を候補なしとして扱ったりしてはならない。

### reference人工テスト

R1からR16を、productionのfixture、牌種定数、正規化関数、期待値定数を
importせずreference側で独立にテストする。次の全項目を期待値と完全一致で
確認する。

- `wait_tiles`
- `wait_tile_count`
- `wait_details`
- `wait_shapes`
- `contains_ryanmen`
- `is_pure_ryanmen`
- `is_multiwait`

特にR15の同一待ち牌に対する複数wait shapeと、R16の異なる未完成断片を
最後まで探索する性質を検査する。

加えて、七対子四枚使いnegative、国士negative、赤5、4枚所有済み、複数暗槓、
必要面子数ゼロをreference単体で検査する。MJAI再生については、生牌の削除、
全副露種、槓後のdoraと嶺上tsumo、成立・未成立リーチ、複数リーチ、本人の
打牌数、河のcall情報、物理行番号を人工イベント列で検査する。

比較adapterには、片側候補の追加・削除、actor、赤牌生表記、手牌、固定面子、
河履歴、wait detail等を意図的に変更し、各差分分類が期待どおりになる故障注入
テストを設ける。手牌順の置換、数牌スートの置換、通常5と赤5の置換による
メタモルフィックテストも行う。

### 実牌譜reference検証段階

必須検証は次の順序で行う。同じ段階では入力source pathを先に決定論的に固定し、
同一入力集合をproduction/referenceの両方へ渡す。

1. R1からR16を含む人工ケース
2. 既に目視監査した2025年の8ファイル
3. 2025年の決定論的な100ファイル
4. 2025年全件
5. 2009年全件
6. 2010年から2024年まで、各年の決定論的な100ファイル

各年の決定論的100ファイルは、次の手順で固定する。

1. 指定年のraw root配下に存在する入力 `.mjson` ファイルを列挙する。
2. raw rootからの相対パスを `/` 区切りへ正規化する。
3. 正規化した相対パス文字列で辞書順に並べる。
4. 対象対局判定を行う前に先頭100ファイルを固定する。
5. production/referenceへ完全に同じ100ファイルを渡す。
6. 各実装がその100ファイル内で `00a9`、`aka_flag is True`、東場を独自に
   判定する。

これは対象対局になったファイルの先頭100件ではない。100件の具体的な一覧を
手作業の正本にはせず、上記選択アルゴリズムを正本とする。再現・監査のため、
実行時に選ばれた相対パス一覧を検証記録へ保存してよい。

目視監査済み2025年の8ファイルは、basenameまたはrelative pathを固定fixture
として保持し、選択結果が実行ごとに変化しないようにする。各段階でcandidate
key集合と全正式比較項目に未説明の不一致がないことを確認し、件数、処理時間、
差分分類別件数を検証記録へ残す。

2009年から2025年までの全件reference比較は、現時点ではIssue #24の必須完了
条件に固定しない。2025年全件と2009年全件の実行後にreference処理速度、年間
実行時間見積り、必要資源を測定する。現実的であれば2009年から2025年までの
全件比較へ拡張する。拡張するか否かの判断理由、測定値、実行範囲、結果を検証
記録へ残す。

実牌譜検証では、カテゴリ別にpositive/negativeケースを決定論的に抽出し、
filenameと各イベント行番号からMJAI原本を目視できるようにする。後に同じactorが
和了した場合は、`hora.pai` の正規化牌種がリーチ時の `wait_tiles` に含まれる
ことも検査する。

### 外部ライブラリと性能

補助検証として、版を固定した `MahjongRepository/mahjong` 等の独立ライブラリで
34牌種を1枚ずつ加えた和了判定を行い、`wait_tiles` を第三者照合してよい。
外部ライブラリは成立リーチ検出、手牌再生、Issue #24固有の `wait_details` の
正本にはしない。現段階では依存へ追加せず、本番分析の必須依存にも含めない。

2025年決定論的1000ファイルのproduction/reference比較が完全一致した後、
productionの `RiichiWaitRecord` factory内で同一手牌の待ちを2回計算していた箇所は、
計算済みの正規結果を内部構築経路へ渡して1回に削減する。公開コンストラクタによる
直接構築では従来どおり再計算し、手牌と待ち情報の整合性検査を維持する。この既知の
重複除去以外のperformance最適化は、必須検証の意味と比較結果を変えないことを前提に、
2025年全件と2009年全件の測定後に別途検討する。

## 出力形式と保存境界

出力形式はこの仕様では確定しない。候補は次の2つとする。

- 1成立リーチ1行のgzip圧縮JSON Lines (`JSONL.gz`)
- 年別に分割したParquet

麻雀ルール、MJAI状態再生、レコード生成はPythonのデータモデルまたは
イテレータを返し、JSONL/Parquetのserializerを呼び出してはならない。
保存処理はレコードを受け取るadapterとして分離する。

両候補は次を満たす必要がある。

- 1成立リーチ1レコードを維持する
- `wait_tiles`、`wait_details`、`actor_discards_before_riichi` の配列・構造を
  情報損失なく保存する
- スキーマバージョンを記録する
- 対象データセット、release tag、対象条件、対象年を記録する
- 年単位でストリーミング生成でき、全件をメモリへ保持しない
- 年別レコード数と整合性検査結果をmanifestへ記録できる
- `data/processed/` 以下へ保存し、生成データをGit管理しない

現時点では `pyarrow` をプロジェクト依存へ追加しない。Parquetを採用する
場合にのみ、保存adapterの実装と合わせて依存追加を別途決定する。

## 完了時の不変条件

- 対象は2009年から2025年の東場である
- 成立リーチ1件につきレコードがちょうど1件ある
- `riichi_discard_number` は本人の宣言牌を含む打牌番号である
- 待ち判定手牌は宣言牌 `dahai` 直後かつ `reach_accepted` 前である
- `reach_accepted` は成立確認だけに使われる
- `wait_tiles` と `wait_details` が待ち情報の正本である
- 同一待ち牌の複数解釈を失わない
- `wait_tile_count == len(wait_tiles)` である
- `is_multiwait == (wait_tile_count >= 3)` である
- 全派生値を正本から再計算できる
- 赤5、暗槓、七対子、国士、多面張、複合形を扱える
- 「良形」は定義されていない
- 後続Issueが河の元情報と `riichi_id` を再利用できる
- 保存形式を変更しても分析ロジックは変わらない
- 不正・不完全MJAIを黙って除外しない
- 人工牌姿と独立reference実装で個別の待ち集合まで検証できる
