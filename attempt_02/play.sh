URL="https://britain-packs-viruses-seven.trycloudflare.com"

for skill in {4..40}; do
  python3 play_match_multiprocess_with_policy.py \
    --games 7 \
    --workers 4 \
    --skill $skill \
    --depth 8 \
    --topk 6 \
    --prob-threshold 0.95 \
    --min-moves 4 \
    --server-url $URL \
    --time 40.0 \
    --output level_${skill}_games.pgn
done
