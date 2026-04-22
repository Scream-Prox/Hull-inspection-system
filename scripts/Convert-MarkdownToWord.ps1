param(
    [Parameter(Mandatory = $true)]
    [string]$SourcePath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$sourceFullPath = [System.IO.Path]::GetFullPath($SourcePath)
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = [System.IO.Path]::GetDirectoryName($outputFullPath)

if (-not (Test-Path -LiteralPath $sourceFullPath)) {
    throw "Source file not found: $sourceFullPath"
}

if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$lines = [System.IO.File]::ReadAllLines($sourceFullPath, [System.Text.Encoding]::UTF8)

$wdStyleNormal = -1
$wdStyleHeading1 = -2
$wdStyleHeading2 = -3
$wdStyleHeading3 = -4
$wdStyleTitle = -63
$wdAlignLeft = 0
$wdAlignCenter = 1
$wdAlignJustify = 3
$wdFormatDocumentDefault = 16

$word = $null
$document = $null

function Add-StyledParagraph {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Selection,
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [int]$StyleId,
        [int]$Alignment = $wdAlignJustify,
        [string]$FontName = "Times New Roman",
        [double]$FontSize = 14,
        [bool]$Bold = $false,
        [bool]$Italic = $false,
        [double]$FirstLineIndent = 35.4,
        [double]$LeftIndent = 0
    )

    $Selection.Style = $StyleId
    $Selection.ParagraphFormat.Alignment = $Alignment
    $Selection.ParagraphFormat.FirstLineIndent = $FirstLineIndent
    $Selection.ParagraphFormat.LeftIndent = $LeftIndent
    $Selection.Font.Name = $FontName
    $Selection.Font.Size = $FontSize
    $Selection.Font.Bold = [int]$Bold
    $Selection.Font.Italic = [int]$Italic
    $Selection.TypeText($Text)
    $Selection.TypeParagraph()
}

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Add()
    $selection = $word.Selection

    $document.Styles.Item($wdStyleNormal).Font.Name = "Times New Roman"
    $document.Styles.Item($wdStyleNormal).Font.Size = 14
    $document.Styles.Item($wdStyleNormal).ParagraphFormat.Alignment = $wdAlignJustify
    $document.Styles.Item($wdStyleNormal).ParagraphFormat.FirstLineIndent = 35.4
    $document.Styles.Item($wdStyleNormal).ParagraphFormat.SpaceAfter = 0
    $document.Styles.Item($wdStyleNormal).ParagraphFormat.SpaceBefore = 0

    $inCodeBlock = $false
    $firstTitleHandled = $false

    foreach ($line in $lines) {
        if ($line -match '^```') {
            $inCodeBlock = -not $inCodeBlock
            $selection.TypeParagraph()
            continue
        }

        if ($inCodeBlock) {
            Add-StyledParagraph -Selection $selection -Text $line -StyleId $wdStyleNormal -Alignment $wdAlignLeft -FontName "Consolas" -FontSize 10 -FirstLineIndent 0 -LeftIndent 28
            continue
        }

        if ($line -match '^\s*$') {
            $selection.TypeParagraph()
            continue
        }

        if ($line -match '^# (.+)$') {
            $text = $matches[1]
            if (-not $firstTitleHandled) {
                Add-StyledParagraph -Selection $selection -Text $text -StyleId $wdStyleTitle -Alignment $wdAlignCenter -FontName "Times New Roman" -FontSize 16 -Bold $true -FirstLineIndent 0
                $firstTitleHandled = $true
            } else {
                Add-StyledParagraph -Selection $selection -Text $text -StyleId $wdStyleHeading1 -Alignment $wdAlignLeft -FontName "Times New Roman" -FontSize 14 -Bold $true -FirstLineIndent 0
            }
            continue
        }

        if ($line -match '^## (.+)$') {
            Add-StyledParagraph -Selection $selection -Text $matches[1] -StyleId $wdStyleHeading1 -Alignment $wdAlignLeft -FontName "Times New Roman" -FontSize 14 -Bold $true -FirstLineIndent 0
            continue
        }

        if ($line -match '^### (.+)$') {
            Add-StyledParagraph -Selection $selection -Text $matches[1] -StyleId $wdStyleHeading2 -Alignment $wdAlignLeft -FontName "Times New Roman" -FontSize 14 -Bold $true -FirstLineIndent 0
            continue
        }

        if ($line -match '^#### (.+)$') {
            Add-StyledParagraph -Selection $selection -Text $matches[1] -StyleId $wdStyleHeading3 -Alignment $wdAlignLeft -FontName "Times New Roman" -FontSize 13 -Bold $true -FirstLineIndent 0
            continue
        }

        if ($line.StartsWith('[') -and $line.EndsWith(']')) {
            Add-StyledParagraph -Selection $selection -Text $line -StyleId $wdStyleNormal -Alignment $wdAlignCenter -FontName "Times New Roman" -FontSize 12 -Italic $true -FirstLineIndent 0
            continue
        }

        if ($line -match '^- (.+)$') {
            Add-StyledParagraph -Selection $selection -Text ("- " + $matches[1]) -StyleId $wdStyleNormal -Alignment $wdAlignJustify -FontName "Times New Roman" -FontSize 14 -FirstLineIndent 0 -LeftIndent 18
            continue
        }

        Add-StyledParagraph -Selection $selection -Text $line -StyleId $wdStyleNormal -Alignment $wdAlignJustify -FontName "Times New Roman" -FontSize 14
    }

    $document.SaveAs([ref]$outputFullPath, [ref]$wdFormatDocumentDefault)
}
finally {
    if ($document -ne $null) {
        $document.Close()
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}

Write-Output "Saved: $outputFullPath"
