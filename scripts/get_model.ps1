# Download the int8 NLLB CTranslate2 model into ./models/nllb-ct2-int8 using curl.
# We use curl (not the HF python client) because the unauthenticated python LFS
# downloader stalls on some Windows networks, while curl streams reliably with resume.
#
#   pwsh -File scripts/get_model.ps1

$ErrorActionPreference = "Stop"
$repo = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
$base = "https://huggingface.co/$repo/resolve/main"
$dir  = Join-Path $PSScriptRoot "..\models\nllb-ct2-int8"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$files = @(
  "config.json", "shared_vocabulary.txt", "sentencepiece.bpe.model",
  "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "model.bin"
)
foreach ($f in $files) {
  Write-Host "downloading $f ..."
  curl.exe -L --retry 5 --retry-delay 3 -C - -o (Join-Path $dir $f) "$base/$f"
}
Write-Host "done -> $dir"
