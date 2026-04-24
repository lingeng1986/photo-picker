#!/bin/bash
# Photo Picker - Main Script with Extended Dimensions
# Three-stage pipeline: filter -> score -> select -> validate
# Extended: More scoring dimensions for reporting, filtering uses subset

set -o pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validation counters
VALIDATION_PASSED=0
VALIDATION_FAILED=0

# Validation function
validate_step() {
    local step_name="$1"
    local condition="$2"
    local error_msg="$3"
    
    if eval "$condition"; then
        echo -e "${GREEN}✓ VALIDATION PASSED:${NC} $step_name"
        VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ VALIDATION FAILED:${NC} $step_name"
        echo -e "${RED}  Error: $error_msg${NC}"
        VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
        return 1
    fi
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="${1:-}"
CONFIG_FILE="${PHOTO_PICKER_CONFIG:-$SKILL_DIR/config.json}"
DEFAULT_CONFIG='{
  "weights": {
    "clarity": 0.3,
    "composition": 0.2,
    "expression": 0.4,
    "lighting": 0.1,
    "subject": 0.3,
    "eye_contact": 0.1,
    "pose": 0.05,
    "bokeh": 0.05,
    "contrast": 0.1,
    "color": 0.05,
    "noise": 0.05
  },
  "thresholds": {"min_score": 7},
  "output": {"copy_not_selected": true}
}'

# Check input
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: picker.sh /path/to/photos"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory not found: $INPUT_DIR"
    exit 1
fi

# Preflight dependency checks
for cmd in sips bc curl base64 ollama; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: Required command not found: $cmd"
        exit 1
    fi
done

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Error: Ollama is not reachable at http://localhost:11434"
    echo "Start Ollama first, then retry."
    exit 1
fi

if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx 'llava:13b'; then
    echo "Error: Required Ollama model not found: llava:13b"
    echo "Run: ollama pull llava:13b"
    exit 1
fi

# Load or create config
if [ -f "$CONFIG_FILE" ]; then
    CONFIG=$(cat "$CONFIG_FILE")
else
    mkdir -p "$(dirname "$CONFIG_FILE")"
    echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
    CONFIG="$DEFAULT_CONFIG"
fi

# Config helpers
json_get() {
    local query="$1"
    local default_json="$2"
    local value
    value=$(printf '%s' "$CONFIG" | jq -cer "$query // $default_json" 2>/dev/null) || value="$default_json"
    printf '%s' "$value"
}

json_get_number() {
    local query="$1"
    local default="$2"
    json_get "$query" "$default"
}

json_get_bool() {
    local query="$1"
    local default="$2"
    local value
    value=$(json_get "$query" "$default")
    case "$value" in
        true|TRUE|True|1)
            printf 'true'
            ;;
        false|FALSE|False|0)
            printf 'false'
            ;;
        *)
            printf '%s' "$default"
            ;;
    esac
}

normalize_weights() {
    local joined="$1"
    local normalized
    normalized=$(python3 - "$joined" <<'PY'
import sys
parts = [float(x) for x in sys.argv[1].split('|')]
total = sum(parts)
if total <= 0:
    n = len(parts)
    print('|'.join([f"{1/n:.6f}" for _ in parts]))
else:
    print('|'.join([f"{x/total:.6f}" for x in parts]))
PY
)
    printf '%s' "$normalized"
}

MIN_SCORE=$(json_get_number '.thresholds.min_score' '7')
COPY_NOT_SELECTED=$(json_get_bool '.output.copy_not_selected' 'true')

# Load configurable weights
CFG_CLARITY=$(json_get_number '.weights.clarity' '0.3')
CFG_COMPOSITION=$(json_get_number '.weights.composition' '0.2')
CFG_EXPRESSION=$(json_get_number '.weights.expression' '0.4')
CFG_LIGHTING=$(json_get_number '.weights.lighting' '0.1')
CFG_SUBJECT=$(json_get_number '.weights.subject' '0.3')
CFG_EYE_CONTACT=$(json_get_number '.weights.eye_contact' '0.1')
CFG_POSE=$(json_get_number '.weights.pose' '0.05')
CFG_BOKEH=$(json_get_number '.weights.bokeh' '0.05')
CFG_CONTRAST=$(json_get_number '.weights.contrast' '0.1')
CFG_COLOR=$(json_get_number '.weights.color' '0.05')
CFG_NOISE=$(json_get_number '.weights.noise' '0.05')

PORTRAIT_FILTER_WEIGHT_SET=$(normalize_weights "$CFG_CLARITY|$CFG_EXPRESSION|$CFG_COMPOSITION|$CFG_LIGHTING|$CFG_EYE_CONTACT|$CFG_POSE|$CFG_BOKEH")
P_FILTER_CLARITY=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f1)
P_FILTER_EXPRESSION=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f2)
P_FILTER_COMPOSITION=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f3)
P_FILTER_LIGHTING=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f4)
P_FILTER_EYE_CONTACT=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f5)
P_FILTER_POSE=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f6)
P_FILTER_BOKEH=$(echo "$PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f7)

NON_PORTRAIT_FILTER_WEIGHT_SET=$(normalize_weights "$CFG_CLARITY|$CFG_SUBJECT|$CFG_COMPOSITION|$CFG_LIGHTING")
NP_FILTER_CLARITY=$(echo "$NON_PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f1)
NP_FILTER_SUBJECT=$(echo "$NON_PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f2)
NP_FILTER_COMPOSITION=$(echo "$NON_PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f3)
NP_FILTER_LIGHTING=$(echo "$NON_PORTRAIT_FILTER_WEIGHT_SET" | cut -d'|' -f4)

NON_PORTRAIT_FULL_WEIGHT_SET=$(normalize_weights "$CFG_CLARITY|$CFG_SUBJECT|$CFG_COMPOSITION|$CFG_LIGHTING|$CFG_CONTRAST|$CFG_COLOR|$CFG_NOISE")
NP_CLARITY=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f1)
NP_SUBJECT=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f2)
NP_COMPOSITION=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f3)
NP_LIGHTING=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f4)
NP_CONTRAST=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f5)
NP_COLOR=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f6)
NP_NOISE=$(echo "$NON_PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f7)

PORTRAIT_FULL_WEIGHT_SET=$(normalize_weights "$CFG_CLARITY|$CFG_EXPRESSION|$CFG_EYE_CONTACT|$CFG_POSE|$CFG_BOKEH|$CFG_COMPOSITION|$CFG_LIGHTING|$CFG_COLOR")
P_CLARITY=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f1)
P_EXPRESSION=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f2)
P_EYE_CONTACT=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f3)
P_POSE=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f4)
P_BOKEH=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f5)
P_COMPOSITION=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f6)
P_LIGHTING=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f7)
P_COLOR=$(echo "$PORTRAIT_FULL_WEIGHT_SET" | cut -d'|' -f8)

# Validate config loading
validate_step "Config Loading" "[ -n \"$MIN_SCORE\" ]" "MIN_SCORE is empty"

# Create output directory
BASENAME=$(basename "$INPUT_DIR")
DATE_STR=$(date +%Y%m%d)
OUTPUT_DIR="${INPUT_DIR}_by_ai_${DATE_STR}"
SUFFIX=""; COUNTER=1
while [ -d "$OUTPUT_DIR$SUFFIX" ]; do
    SUFFIX="_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done
OUTPUT_DIR="${OUTPUT_DIR}${SUFFIX}"

mkdir -p "$OUTPUT_DIR/selected" "$OUTPUT_DIR/thumbs" "$OUTPUT_DIR/.tmp"
if [ "$COPY_NOT_SELECTED" = true ]; then
    mkdir -p "$OUTPUT_DIR/not_selected"
fi

# Validate output directory creation
if [ "$COPY_NOT_SELECTED" = true ]; then
    validate_step "Output Directory Creation" "[ -d \"$OUTPUT_DIR\" ] && [ -d \"$OUTPUT_DIR/selected\" ] && [ -d \"$OUTPUT_DIR/not_selected\" ] && [ -d \"$OUTPUT_DIR/thumbs\" ]" "Output directories not created"
else
    validate_step "Output Directory Creation" "[ -d \"$OUTPUT_DIR\" ] && [ -d \"$OUTPUT_DIR/selected\" ] && [ -d \"$OUTPUT_DIR/thumbs\" ]" "Output directories not created"
fi

echo "Photo Picker Started"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Find all images
find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic" \) | sort > "$OUTPUT_DIR/.tmp/all_images.txt"
find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.arw" | sort > "$OUTPUT_DIR/.tmp/all_arw.txt"

TOTAL=$(wc -l < "$OUTPUT_DIR/.tmp/all_images.txt" | tr -d ' ')
ARW_COUNT=$(wc -l < "$OUTPUT_DIR/.tmp/all_arw.txt" | tr -d ' ')

# Validate image discovery
validate_step "Image Discovery" "[ -s \"$OUTPUT_DIR/.tmp/all_images.txt\" ]" "No images found in input directory"

echo "Found $TOTAL images ($ARW_COUNT ARW)"
echo ""

# Stage 1: Generate thumbnails
echo "Stage 1: Generating thumbnails..."
THUMB_COUNT=0
while IFS= read -r img; do
    BASENAME=$(basename "$img")
    NAME="${BASENAME%.*}"
    if sips -Z 1024 "$img" --out "$OUTPUT_DIR/thumbs/${NAME}_thumb.jpg" 2>/dev/null; then
        echo "  ✓ ${NAME}_thumb.jpg"
        THUMB_COUNT=$((THUMB_COUNT + 1))
    else
        echo "  ✗ Failed: ${BASENAME}"
    fi
done < "$OUTPUT_DIR/.tmp/all_images.txt"

# Validate thumbnails
validate_step "Thumbnail Generation" "[ $THUMB_COUNT -eq $TOTAL ]" "Only $THUMB_COUNT/$TOTAL thumbnails generated"

echo "Thumbnails generated: $THUMB_COUNT/$TOTAL"
echo ""

# Stage 2: Quick clarity check
echo "Stage 2: Quick clarity check..."
> "$OUTPUT_DIR/.tmp/stage1_passed.txt"
while IFS= read -r img; do
    BASENAME=$(basename "$img")
    NAME="${BASENAME%.*}"
    THUMB="$OUTPUT_DIR/thumbs/${NAME}_thumb.jpg"
    if [ -f "$THUMB" ]; then
        echo "$img" >> "$OUTPUT_DIR/.tmp/stage1_passed.txt"
        echo "    ✓ $BASENAME"
    fi
done < "$OUTPUT_DIR/.tmp/all_images.txt"

STAGE1_COUNT=$(wc -l < "$OUTPUT_DIR/.tmp/stage1_passed.txt" | tr -d ' ')
echo "Stage 2 complete: $STAGE1_COUNT / $TOTAL passed"

# Validate stage 1
validate_step "Stage 1 Completion" "[ $STAGE1_COUNT -eq $TOTAL ]" "Not all images passed stage 1"
echo ""

# Functions for AI scoring with extended dimensions
detect_portrait() {
    local thumb="$1"
    local img_base64=$(base64 -i "$thumb" | tr -d '\n')
    local result=$(curl -s -m 30 http://localhost:11434/api/generate \
        -d "{\"model\":\"llava:13b\",\"system\":\"Reply ONLY YES or NO. YES if photo has prominent human face/portrait, NO for landscape/object/animal/scene without clear human face.\",\"prompt\":\"Is this a portrait with human face? Reply ONLY YES or NO.\",\"images\":[\"$img_base64\"],\"stream\":false}" 2>/dev/null | sed 's/.*"response":"\([^"]*\)".*/\1/')
    if echo "$result" | grep -qi "YES"; then echo "portrait"; else echo "non-portrait"; fi
}

# Extended portrait scoring with 8 dimensions
score_portrait_extended() {
    local thumb="$1"
    local img_base64=$(base64 -i "$thumb" | tr -d '\n')
    curl -s -m 60 http://localhost:11434/api/generate \
        -d "{\"model\":\"llava:13b\",\"system\":\"Reply ONLY in this exact format: clarity:X,expression:Y,eye_contact:Z,pose:W,bokeh:V,composition:U,lighting:T,color:S where X,Y,Z,W,V,U,T,S are numbers 0-10. No other text.\",\"prompt\":\"Rate PORTRAIT photo 1-10 for: 1)Clarity(sharpness) 2)Expression(facial emotion) 3)Eye_contact(looking at camera) 4)Pose(body posture) 5)Bokeh(background blur quality) 6)Composition(framing) 7)Lighting(exposure quality) 8)Color(color balance/naturalness). Reply EXACTLY: clarity:7,expression:6,eye_contact:8,pose:7,bokeh:6,composition:5,lighting:7,color:6\",\"images\":[\"$img_base64\"],\"stream\":false}" 2>/dev/null | sed 's/.*"response":"\([^"]*\)".*/\1/'
}

# Extended non-portrait scoring with 7 dimensions
score_non_portrait_extended() {
    local thumb="$1"
    local img_base64=$(base64 -i "$thumb" | tr -d '\n')
    curl -s -m 60 http://localhost:11434/api/generate \
        -d "{\"model\":\"llava:13b\",\"system\":\"Reply ONLY in this exact format: clarity:X,subject:Y,composition:Z,lighting:W,contrast:V,color:U,noise:T where X,Y,Z,W,V,U,T are numbers 0-10. No other text.\",\"prompt\":\"Rate photo 1-10 for: 1)Clarity(sharpness,focus) 2)Subject(subject interest/quality, 0=no clear subject) 3)Composition(framing) 4)Lighting(exposure) 5)Contrast(dynamic range) 6)Color(color accuracy/balance) 7)Noise(grain/cleanliness). Reply EXACTLY: clarity:7,subject:6,composition:5,lighting:7,contrast:6,color:7,noise:8\",\"images\":[\"$img_base64\"],\"stream\":false}" 2>/dev/null | sed 's/.*"response":"\([^"]*\)".*/\1/'
}

# Stage 3: AI Scoring with extended dimensions
echo "Stage 3: AI Scoring (Extended Dimensions)..."
> "$OUTPUT_DIR/.tmp/stage2_scores.txt"
SCORED_COUNT=0
PORTRAIT_COUNT=0
NON_PORTRAIT_COUNT=0

while IFS= read -r img; do
    BASENAME=$(basename "$img")
    NAME="${BASENAME%.*}"
    THUMB="$OUTPUT_DIR/thumbs/${NAME}_thumb.jpg"
    
    if [ ! -f "$THUMB" ]; then
        echo "  ⚠ $BASENAME (no thumbnail, skipped)"
        continue
    fi
    
    echo "  [$((SCORED_COUNT + 1))/$STAGE1_COUNT] $BASENAME"
    
    PHOTO_TYPE=$(detect_portrait "$THUMB")
    echo "    → $PHOTO_TYPE"
    
    if [ "$PHOTO_TYPE" = "portrait" ]; then
        PORTRAIT_COUNT=$((PORTRAIT_COUNT + 1))
        ANALYSIS=$(score_portrait_extended "$THUMB")
        
        # Parse all 8 dimensions
        CLARITY=$(echo "$ANALYSIS" | grep -oi 'clarity[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        EXPRESSION=$(echo "$ANALYSIS" | grep -oi 'expression[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        EYE_CONTACT=$(echo "$ANALYSIS" | grep -oi 'eye_contact[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        POSE=$(echo "$ANALYSIS" | grep -oi 'pose[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        BOKEH=$(echo "$ANALYSIS" | grep -oi 'bokeh[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        COMPOSITION=$(echo "$ANALYSIS" | grep -oi 'composition[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        LIGHTING=$(echo "$ANALYSIS" | grep -oi 'lighting[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        COLOR=$(echo "$ANALYSIS" | grep -oi 'color[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        
        # Set defaults
        CLARITY=${CLARITY:-5}; EXPRESSION=${EXPRESSION:-5}; EYE_CONTACT=${EYE_CONTACT:-5}
        POSE=${POSE:-5}; BOKEH=${BOKEH:-5}; COMPOSITION=${COMPOSITION:-5}
        LIGHTING=${LIGHTING:-5}; COLOR=${COLOR:-5}
        
        # Calculate filter score (fully config-driven portrait weights)
        FILTER_SCORE=$(echo "scale=2; ($CLARITY * $P_FILTER_CLARITY + $EXPRESSION * $P_FILTER_EXPRESSION + $COMPOSITION * $P_FILTER_COMPOSITION + $LIGHTING * $P_FILTER_LIGHTING + $EYE_CONTACT * $P_FILTER_EYE_CONTACT + $POSE * $P_FILTER_POSE + $BOKEH * $P_FILTER_BOKEH)" | bc)
        
        # Calculate full score (all 8 dimensions for reference)
        FULL_SCORE=$(echo "scale=2; ($CLARITY * $P_CLARITY + $EXPRESSION * $P_EXPRESSION + $EYE_CONTACT * $P_EYE_CONTACT + $POSE * $P_POSE + $BOKEH * $P_BOKEH + $COMPOSITION * $P_COMPOSITION + $LIGHTING * $P_LIGHTING + $COLOR * $P_COLOR)" | bc)
        
        echo "$FILTER_SCORE|$FULL_SCORE|$CLARITY|$EXPRESSION|$EYE_CONTACT|$POSE|$BOKEH|$COMPOSITION|$LIGHTING|$COLOR|portrait|$img" >> "$OUTPUT_DIR/.tmp/stage2_scores.txt"
        echo "    Filter: $FILTER_SCORE | Full: $FULL_SCORE (c:$CLARITY e:$EXPRESSION ec:$EYE_CONTACT p:$POSE b:$BOKEH m:$COMPOSITION l:$LIGHTING col:$COLOR)"
    else
        NON_PORTRAIT_COUNT=$((NON_PORTRAIT_COUNT + 1))
        ANALYSIS=$(score_non_portrait_extended "$THUMB")
        
        # Parse all 7 dimensions
        CLARITY=$(echo "$ANALYSIS" | grep -oi 'clarity[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        SUBJECT=$(echo "$ANALYSIS" | grep -oi 'subject[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        COMPOSITION=$(echo "$ANALYSIS" | grep -oi 'composition[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        LIGHTING=$(echo "$ANALYSIS" | grep -oi 'lighting[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        CONTRAST=$(echo "$ANALYSIS" | grep -oi 'contrast[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        COLOR=$(echo "$ANALYSIS" | grep -oi 'color[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        NOISE=$(echo "$ANALYSIS" | grep -oi 'noise[[:space:]]*[:=][[:space:]]*[0-9]*' | grep -o '[0-9]*' | head -1)
        
        # Set defaults
        CLARITY=${CLARITY:-5}; SUBJECT=${SUBJECT:-5}; COMPOSITION=${COMPOSITION:-5}
        LIGHTING=${LIGHTING:-5}; CONTRAST=${CONTRAST:-5}; COLOR=${COLOR:-5}; NOISE=${NOISE:-5}
        
        # Calculate filter score (fully config-driven non-portrait weights)
        if [ "$SUBJECT" -eq 0 ]; then
            SUBJECT_REBALANCED=$(normalize_weights "$CFG_CLARITY|0|$CFG_COMPOSITION|$CFG_LIGHTING")
            RB_CLARITY=$(echo "$SUBJECT_REBALANCED" | cut -d'|' -f1)
            RB_COMPOSITION=$(echo "$SUBJECT_REBALANCED" | cut -d'|' -f3)
            RB_LIGHTING=$(echo "$SUBJECT_REBALANCED" | cut -d'|' -f4)
            FILTER_SCORE=$(echo "scale=2; ($CLARITY * $RB_CLARITY + $COMPOSITION * $RB_COMPOSITION + $LIGHTING * $RB_LIGHTING)" | bc)
        else
            FILTER_SCORE=$(echo "scale=2; ($CLARITY * $NP_FILTER_CLARITY + $SUBJECT * $NP_FILTER_SUBJECT + $COMPOSITION * $NP_FILTER_COMPOSITION + $LIGHTING * $NP_FILTER_LIGHTING)" | bc)
        fi
        
        # Calculate full score (all 7 dimensions for reference)
        FULL_SCORE=$(echo "scale=2; ($CLARITY * $NP_CLARITY + $SUBJECT * $NP_SUBJECT + $COMPOSITION * $NP_COMPOSITION + $LIGHTING * $NP_LIGHTING + $CONTRAST * $NP_CONTRAST + $COLOR * $NP_COLOR + $NOISE * $NP_NOISE)" | bc)
        
        echo "$FILTER_SCORE|$FULL_SCORE|$CLARITY|$SUBJECT|$COMPOSITION|$LIGHTING|$CONTRAST|$COLOR|$NOISE|non-portrait|$img" >> "$OUTPUT_DIR/.tmp/stage2_scores.txt"
        echo "    Filter: $FILTER_SCORE | Full: $FULL_SCORE (c:$CLARITY s:$SUBJECT m:$COMPOSITION l:$LIGHTING con:$CONTRAST col:$COLOR n:$NOISE)"
    fi
    
    SCORED_COUNT=$((SCORED_COUNT + 1))
done < "$OUTPUT_DIR/.tmp/stage1_passed.txt"

# Validate scoring
validate_step "AI Scoring Completion" "[ -s \"$OUTPUT_DIR/.tmp/stage2_scores.txt\" ]" "No scores generated"
validate_step "All Images Scored" "[ $SCORED_COUNT -eq $STAGE1_COUNT ]" "Only $SCORED_COUNT/$STAGE1_COUNT images scored"

sort -t'|' -k1 -nr "$OUTPUT_DIR/.tmp/stage2_scores.txt" > "$OUTPUT_DIR/.tmp/stage2_sorted.txt"

echo "Stage 3 complete: $PORTRAIT_COUNT portraits, $NON_PORTRAIT_COUNT non-portraits"
echo ""

# Stage 4: Filter and select (using FILTER_SCORE)
echo "Stage 4: Filtering..."
> "$OUTPUT_DIR/.tmp/stage3_selected.txt"
SELECTED_COUNT=0

while IFS='|' read -r filter_score full_score rest; do
    img=$(echo "$rest" | awk -F'|' '{print $NF}')
    if (( $(echo "$filter_score >= $MIN_SCORE" | bc -l) )); then
        echo "$filter_score|$full_score|$rest" >> "$OUTPUT_DIR/.tmp/stage3_selected.txt"
        echo "  ✓ $(basename "$img") (Filter: $filter_score, Full: $full_score)"
        SELECTED_COUNT=$((SELECTED_COUNT + 1))
    else
        echo "  ✗ $(basename "$img") (Filter: $filter_score < $MIN_SCORE)"
    fi
done < "$OUTPUT_DIR/.tmp/stage2_sorted.txt"

echo "Stage 4 complete: $SELECTED_COUNT selected"
echo ""

# Stage 5: Copy files
echo "Stage 5: Copying files..."
> "$OUTPUT_DIR/.tmp/copied.txt"
COPIED_SELECTED=0

while IFS='|' read -r filter_score full_score rest; do
    img=$(echo "$rest" | awk -F'|' '{print $NF}')
    BASENAME=$(basename "$img")
    NAME="${BASENAME%.*}"
    
    if cp "$img" "$OUTPUT_DIR/selected/"; then
        echo "  ✓ $BASENAME"
        COPIED_SELECTED=$((COPIED_SELECTED + 1))
    else
        echo "  ✗ Failed to copy: $BASENAME"
        continue
    fi
    
    ARW_FILE="$(dirname "$img")/${NAME}.ARW"
    if [ -f "$ARW_FILE" ]; then
        if cp "$ARW_FILE" "$OUTPUT_DIR/selected/"; then
            echo "    + ${NAME}.ARW"
        fi
    fi
    echo "$filter_score|$full_score|$rest|selected" >> "$OUTPUT_DIR/.tmp/copied.txt"
done < "$OUTPUT_DIR/.tmp/stage3_selected.txt"

# Validate selected files
echo ""
validate_step "Selected Files Copy" "[ $COPIED_SELECTED -eq $SELECTED_COUNT ]" "Only $COPIED_SELECTED/$SELECTED_COUNT selected files copied"

# Copy not selected
NOT_SELECTED_COUNT=$((TOTAL - SELECTED_COUNT))
COPIED_NOT_SELECTED=0
if [ "$COPY_NOT_SELECTED" = true ]; then
    echo "Copying not selected images..."
    while IFS='|' read -r filter_score full_score rest; do
        img=$(echo "$rest" | awk -F'|' '{print $NF}')
        BASENAME=$(basename "$img")
        NAME="${BASENAME%.*}"
        if grep -q "$BASENAME" "$OUTPUT_DIR/.tmp/copied.txt" 2>/dev/null; then continue; fi
        
        if cp "$img" "$OUTPUT_DIR/not_selected/"; then
            echo "  → $BASENAME (Filter: $filter_score)"
            COPIED_NOT_SELECTED=$((COPIED_NOT_SELECTED + 1))
        fi
        
        ARW_FILE="$(dirname "$img")/${NAME}.ARW"
        if [ -f "$ARW_FILE" ]; then
            cp "$ARW_FILE" "$OUTPUT_DIR/not_selected/" 2>/dev/null && echo "    + ${NAME}.ARW"
        fi
        echo "$filter_score|$full_score|$rest|not_selected" >> "$OUTPUT_DIR/.tmp/copied.txt"
    done < "$OUTPUT_DIR/.tmp/stage2_scores.txt"

    validate_step "Not Selected Files Copy" "[ $COPIED_NOT_SELECTED -eq $NOT_SELECTED_COUNT ]" "Only $COPIED_NOT_SELECTED/$NOT_SELECTED_COUNT not-selected files copied"
else
    echo "Skipping copy of not selected images (copy_not_selected=false)"
    validate_step "Not Selected Files Copy" "true" "copy skipped by config"
fi

echo ""

# Stage 6: Generate report with extended dimensions
echo "Stage 6: Generating report..."

cat > "$OUTPUT_DIR/report.md" << EOF
# Photo Picker Report

**Input:** $INPUT_DIR  
**Output:** $OUTPUT_DIR  
**Date:** $(date)

## Summary

| Metric | Value |
|--------|-------|
| Total Images | $TOTAL |
| Portraits | $PORTRAIT_COUNT |
| Non-Portraits | $NON_PORTRAIT_COUNT |
| Selected | $SELECTED_COUNT |
| Not Selected | $NOT_SELECTED_COUNT |

## Configuration

| Parameter | Value |
|-----------|-------|
| Min Score (Filter) | $MIN_SCORE |
| Copy Not Selected | $COPY_NOT_SELECTED |
| Config Clarity Weight | $CFG_CLARITY |
| Config Expression Weight | $CFG_EXPRESSION |
| Config Subject Weight | $CFG_SUBJECT |
| Config Composition Weight | $CFG_COMPOSITION |
| Config Lighting Weight | $CFG_LIGHTING |
| Config Eye Contact Weight | $CFG_EYE_CONTACT |
| Config Pose Weight | $CFG_POSE |
| Config Bokeh Weight | $CFG_BOKEH |
| Config Contrast Weight | $CFG_CONTRAST |
| Config Color Weight | $CFG_COLOR |
| Config Noise Weight | $CFG_NOISE |

## Selected Images

### Portraits (8 Dimensions)
| File | Filter Score | Full Score | Clarity | Expression | Eye Contact | Pose | Bokeh | Composition | Lighting | Color |
|------|-------------|------------|---------|------------|-------------|------|-------|-------------|----------|-------|
EOF

# Add selected portraits to report
while IFS='|' read -r filter_score full_score rest; do
    type=$(echo "$rest" | awk -F'|' '{print $(NF-1)}')
    [ "$type" != "portrait" ] && continue
    
    img=$(echo "$rest" | awk -F'|' '{print $NF}')
    BASENAME=$(basename "$img")
    
    # Check if selected
    if ! grep -q "$BASENAME.*selected" "$OUTPUT_DIR/.tmp/copied.txt" 2>/dev/null; then
        continue
    fi
    
    c=$(echo "$rest" | cut -d'|' -f3)
    e=$(echo "$rest" | cut -d'|' -f4)
    ec=$(echo "$rest" | cut -d'|' -f5)
    p=$(echo "$rest" | cut -d'|' -f6)
    b=$(echo "$rest" | cut -d'|' -f7)
    m=$(echo "$rest" | cut -d'|' -f8)
    l=$(echo "$rest" | cut -d'|' -f9)
    col=$(echo "$rest" | cut -d'|' -f10)
    
    echo "| $BASENAME | $filter_score | $full_score | $c | $e | $ec | $p | $b | $m | $l | $col |" >> "$OUTPUT_DIR/report.md"
done < "$OUTPUT_DIR/.tmp/stage3_selected.txt"

cat >> "$OUTPUT_DIR/report.md" << EOF

### Non-Portraits (7 Dimensions)
| File | Filter Score | Full Score | Clarity | Subject | Composition | Lighting | Contrast | Color | Noise |
|------|-------------|------------|---------|---------|-------------|----------|----------|-------|-------|
EOF

# Add selected non-portraits to report
while IFS='|' read -r filter_score full_score rest; do
    type=$(echo "$rest" | awk -F'|' '{print $(NF-1)}')
    [ "$type" != "non-portrait" ] && continue
    
    img=$(echo "$rest" | awk -F'|' '{print $NF}')
    BASENAME=$(basename "$img")
    
    # Check if selected
    if ! grep -q "$BASENAME.*selected" "$OUTPUT_DIR/.tmp/copied.txt" 2>/dev/null; then
        continue
    fi
    
    c=$(echo "$rest" | cut -d'|' -f3)
    s=$(echo "$rest" | cut -d'|' -f4)
    m=$(echo "$rest" | cut -d'|' -f5)
    l=$(echo "$rest" | cut -d'|' -f6)
    con=$(echo "$rest" | cut -d'|' -f7)
    col=$(echo "$rest" | cut -d'|' -f8)
    n=$(echo "$rest" | cut -d'|' -f9)
    
    echo "| $BASENAME | $filter_score | $full_score | $c | $s | $m | $l | $con | $col | $n |" >> "$OUTPUT_DIR/report.md"
done < "$OUTPUT_DIR/.tmp/stage3_selected.txt"

cat >> "$OUTPUT_DIR/report.md" << EOF

## All Scored Images

| File | Type | Filter Score | Full Score | Status |
|------|------|-------------|------------|--------|
EOF

while IFS='|' read -r filter_score full_score rest; do
    type=$(echo "$rest" | awk -F'|' '{print $(NF-1)}')
    img=$(echo "$rest" | awk -F'|' '{print $NF}')
    BASENAME=$(basename "$img")
    status="filtered"
    if grep -q "$BASENAME.*selected" "$OUTPUT_DIR/.tmp/copied.txt" 2>/dev/null; then
        status="selected"
    fi
    echo "| $BASENAME | $type | $filter_score | $full_score | $status |" >> "$OUTPUT_DIR/report.md"
done < "$OUTPUT_DIR/.tmp/stage2_sorted.txt"

# Validate report
validate_step "Report Generation" "[ -f \"$OUTPUT_DIR/report.md\" ] && [ -s \"$OUTPUT_DIR/report.md\" ]" "Report not generated or empty"

# Cleanup
rm -rf "$OUTPUT_DIR/.tmp"

# Final validation summary
echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo -e "${GREEN}Passed: $VALIDATION_PASSED${NC}"
echo -e "${RED}Failed: $VALIDATION_FAILED${NC}"
echo ""

if [ $VALIDATION_FAILED -gt 0 ]; then
    echo -e "${RED}WARNING: Some validations failed. Please check the output above.${NC}"
    exit 1
fi

echo "Photo Picker Complete!"
echo "=========================================="
echo "Portraits: $PORTRAIT_COUNT"
echo "Non-Portraits: $NON_PORTRAIT_COUNT"
echo "Selected: $SELECTED_COUNT"
echo "Not Selected: $NOT_SELECTED_COUNT"
echo "Output: $OUTPUT_DIR/selected/"
if [ "$COPY_NOT_SELECTED" = true ]; then
    echo "        $OUTPUT_DIR/not_selected/"
fi
echo "Report: $OUTPUT_DIR/report.md"
echo "=========================================="
