Add-Type -AssemblyName System.Drawing

$outputPath = Join-Path $PSScriptRoot '..\docs\figures\combo-information-flow.png'
$outputPath = [System.IO.Path]::GetFullPath($outputPath)
$outputDirectory = Split-Path -Parent $outputPath
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null

$bitmap = [System.Drawing.Bitmap]::new(1280, 720)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.Color]::FromArgb(247, 245, 239))

$dark = [System.Drawing.Color]::FromArgb(32, 36, 42)
$muted = [System.Drawing.Color]::FromArgb(92, 92, 92)
$border = [System.Drawing.Color]::FromArgb(210, 205, 194)
$white = [System.Drawing.Color]::White
$blue = [System.Drawing.Color]::FromArgb(229, 241, 255)
$red = [System.Drawing.Color]::FromArgb(151, 55, 43)

$fontTitle = [System.Drawing.Font]::new('Yu Gothic UI', 28, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = [System.Drawing.Font]::new('Yu Gothic UI', 17)
$fontTile = [System.Drawing.Font]::new('Yu Gothic UI', 25, [System.Drawing.FontStyle]::Bold)
$fontBody = [System.Drawing.Font]::new('Yu Gothic UI', 17)
$fontBodyBold = [System.Drawing.Font]::new('Yu Gothic UI', 17, [System.Drawing.FontStyle]::Bold)
$fontSmall = [System.Drawing.Font]::new('Yu Gothic UI', 14)
$fontTotal = [System.Drawing.Font]::new('Yu Gothic UI', 22, [System.Drawing.FontStyle]::Bold)
$fontConclusion = [System.Drawing.Font]::new('Yu Gothic UI', 20, [System.Drawing.FontStyle]::Bold)

$brushDark = [System.Drawing.SolidBrush]::new($dark)
$brushMuted = [System.Drawing.SolidBrush]::new($muted)
$brushWhite = [System.Drawing.SolidBrush]::new($white)
$brushBlue = [System.Drawing.SolidBrush]::new($blue)
$brushRed = [System.Drawing.SolidBrush]::new($red)
$penBorder = [System.Drawing.Pen]::new($border, 2)
$penArrow = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(86, 97, 122), 4)
$penArrow.EndCap = [System.Drawing.Drawing2D.LineCap]::ArrowAnchor

function Draw-CenteredText {
    param(
        [System.Drawing.Graphics]$Canvas,
        [string]$Text,
        [System.Drawing.Font]$Font,
        [System.Drawing.Brush]$Brush,
        [System.Drawing.RectangleF]$Rect
    )
    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $Canvas.DrawString($Text, $Font, $Brush, $Rect, $format)
    $format.Dispose()
}

function Draw-Box {
    param(
        [System.Drawing.Graphics]$Canvas,
        [System.Drawing.RectangleF]$Rect,
        [System.Drawing.Brush]$Fill,
        [System.Drawing.Pen]$Outline
    )
    $Canvas.FillRectangle($Fill, $Rect)
    $Canvas.DrawRectangle($Outline, $Rect.X, $Rect.Y, $Rect.Width, $Rect.Height)
}

$graphics.DrawString('コンボ数は、見えている枚数の「言い換え」', $fontTitle, $brushDark, 54, 28)
$graphics.DrawString('例：7pが当たりになる形を数える（残り枚数は説明用の仮定）', $fontSubtitle, $brushMuted, 57, 80)

$tileXs = @(245, 405, 565, 725, 885)
$tileNames = @('5p', '6p', '7p', '8p', '9p')
$tileCounts = @('残り2枚', '残り3枚', '残り3枚', '残り2枚', '残り3枚')
for ($i = 0; $i -lt $tileXs.Count; $i++) {
    $tileRect = [System.Drawing.RectangleF]::new($tileXs[$i], 125, 125, 100)
    Draw-Box $graphics $tileRect $brushWhite $penBorder
    Draw-CenteredText $graphics $tileNames[$i] $fontTile $brushDark ([System.Drawing.RectangleF]::new($tileXs[$i], 128, 125, 55))
    Draw-CenteredText $graphics $tileCounts[$i] $fontSmall $brushMuted ([System.Drawing.RectangleF]::new($tileXs[$i], 180, 125, 38))
}

$leftPanel = [System.Drawing.RectangleF]::new(65, 255, 700, 210)
$rightPanel = [System.Drawing.RectangleF]::new(790, 255, 425, 210)
Draw-Box $graphics $leftPanel $brushWhite $penBorder
Draw-Box $graphics $rightPanel $brushWhite $penBorder
$graphics.DrawString('7pを使う順子', $fontBodyBold, $brushDark, 90, 274)
$graphics.DrawString('5p・6p・7p', $fontBody, $brushDark, 105, 318)
$graphics.DrawString('2 × 3 ＝ 6通り', $fontBodyBold, $brushDark, 480, 318)
$graphics.DrawString('6p・7p・8p', $fontBody, $brushDark, 105, 365)
$graphics.DrawString('3 × 2 ＝ 6通り', $fontBodyBold, $brushDark, 480, 365)
$graphics.DrawString('7p・8p・9p', $fontBody, $brushDark, 105, 412)
$graphics.DrawString('2 × 3 ＝ 6通り', $fontBodyBold, $brushDark, 480, 412)

$graphics.DrawString('そのほかの待ち方', $fontBodyBold, $brushDark, 815, 274)
$graphics.DrawString('7pの対子', $fontBody, $brushDark, 830, 335)
$graphics.DrawString('3通り', $fontBodyBold, $brushDark, 1080, 335)
$graphics.DrawString('7p単騎', $fontBody, $brushDark, 830, 392)
$graphics.DrawString('3通り', $fontBodyBold, $brushDark, 1080, 392)

$totalRect = [System.Drawing.RectangleF]::new(220, 485, 840, 62)
$graphics.FillRectangle($brushBlue, $totalRect)
Draw-CenteredText $graphics '6 ＋ 6 ＋ 6 ＋ 3 ＋ 3 ＝ コンボ数24' $fontTotal $brushDark $totalRect

$sourceRect = [System.Drawing.RectangleF]::new(65, 575, 325, 72)
$targetTopRect = [System.Drawing.RectangleF]::new(535, 557, 660, 44)
$targetBottomRect = [System.Drawing.RectangleF]::new(535, 615, 660, 44)
Draw-Box $graphics $sourceRect $brushWhite $penBorder
Draw-Box $graphics $targetTopRect $brushWhite $penBorder
Draw-Box $graphics $targetBottomRect $brushWhite $penBorder
Draw-CenteredText $graphics "場と手牌から分かる`n周辺牌の残り枚数" $fontBodyBold $brushDark $sourceRect
Draw-CenteredText $graphics 'そのまま使って危険度を予測' $fontBody $brushDark $targetTopRect
Draw-CenteredText $graphics '掛け算してコンボ数にしてから予測' $fontBody $brushDark $targetBottomRect
$graphics.DrawLine($penArrow, 405, 595, 520, 579)
$graphics.DrawLine($penArrow, 405, 625, 520, 637)

$conclusionRect = [System.Drawing.RectangleF]::new(145, 670, 990, 38)
Draw-CenteredText $graphics 'どちらも元は同じ情報。コンボ数を足しても、新しい牌の情報は増えない。' $fontConclusion $brushRed $conclusionRect

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$graphics.Dispose()
$bitmap.Dispose()
$fontTitle.Dispose()
$fontSubtitle.Dispose()
$fontTile.Dispose()
$fontBody.Dispose()
$fontBodyBold.Dispose()
$fontSmall.Dispose()
$fontTotal.Dispose()
$fontConclusion.Dispose()
$brushDark.Dispose()
$brushMuted.Dispose()
$brushWhite.Dispose()
$brushBlue.Dispose()
$brushRed.Dispose()
$penBorder.Dispose()
$penArrow.Dispose()

Write-Output $outputPath
