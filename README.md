# =========================
# ONE-SHOT SETUP + RUN (PowerShell)
# Run this at repo root: ...\LJC_Stripformer
# =========================

# (0) User settings: ONLY edit these 3 paths
$ENV_NAME   = "Stripformer"
$GOPRO_ROOT = "D:\datasets\GOPRO"     # <- 너의 GoPro 경로로 수정
$HIDE_ROOT  = "D:\datasets\HIDE"      # <- 너의 HIDE 경로로 수정
$WEIGHTS    = ".\Pre_trained_model\Stripformer_gopro.pth"  # 또는 공식 weights 파일 경로로 수정

# (1) Sanity check: are we at repo root?
if (!(Test-Path ".\requirements.txt")) { throw "requirements.txt not found. Run this in repo root folder." }

# (2) Create conda env if not exists
$envList = (conda env list) 2>$null
if ($LASTEXITCODE -ne 0) { throw "conda not available in this terminal. (Try conda hook / init first.)" }

if ($envList -notmatch "^\s*$ENV_NAME\s") {
  Write-Host "[1/6] Creating conda env: $ENV_NAME"
  conda create -n $ENV_NAME python=3.8 -y
} else {
  Write-Host "[1/6] Conda env exists: $ENV_NAME"
}

# (3) Install deps (pip) inside env (no need to 'conda activate')
Write-Host "[2/6] Installing dependencies (pip install -r requirements.txt)"
conda run -n $ENV_NAME python -m pip install --upgrade pip
conda run -n $ENV_NAME python -m pip install -r .\requirements.txt

# (4) Quick checks
Write-Host "[3/6] Python path in env:"
conda run -n $ENV_NAME python -c "import sys; print(sys.executable)"

# (5) (Optional) show dataset & weight paths
Write-Host "[4/6] Paths"
Write-Host "  GOPRO_ROOT = $GOPRO_ROOT"
Write-Host "  HIDE_ROOT  = $HIDE_ROOT"
Write-Host "  WEIGHTS    = $WEIGHTS"
if (!(Test-Path $WEIGHTS)) {
  Write-Host "  [WARN] weights file not found at: $WEIGHTS"
  Write-Host "  -> Download official Stripformer_gopro.pth (or use your own) then set `$WEIGHTS correctly."
}

# (6) Run: choose what you want (uncomment only what you need)

# ---- A) TRAIN (your repo entry)
# If your training script reads config internally, this may work as-is:
# Write-Host "[5/6] Training (cross-att variant)"
# conda run -n $ENV_NAME python .\train_Stripformer_cross_att_t2.py

# If your training script supports --config:
# Write-Host "[5/6] Training with config"
# conda run -n $ENV_NAME python .\train_Stripformer_cross_att_t2.py --config .\config\config_Stripformer_gopro.yaml

# ---- B) TEST (your repo entry)
# Write-Host "[5/6] Testing on GoPro"
# conda run -n $ENV_NAME python .\test_Stripformer_gopro.py --config .\config\config_Stripformer_gopro.yaml

# ---- C) PREDICT / EXPORT (official-style usage in similar Stripformer repos) :contentReference[oaicite:1]{index=1}
Write-Host "[5/6] Predict GoPro test results (requires weights + dataset path set in your code/config)"
conda run -n $ENV_NAME python .\predict_GoPro_test_results.py --weights_path $WEIGHTS

# If you also have HIDE predict script in your repo later:
# Write-Host "[5/6] Predict HIDE test results"
# conda run -n $ENV_NAME python .\predict_HIDE_results.py --weights_path $WEIGHTS

Write-Host "[6/6] Done."
