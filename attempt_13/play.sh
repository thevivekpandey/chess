URL="https://hopkins-arrive-succeed-routes.trycloudflare.com"

python3 play_match_multiprocess_with_policy.py \
  --games 10 \
  --workers 3 \
  --start-skill 4 \
  --end-skill 7 \
  --depth 7 \
  --topk 6 \
  --prob-threshold 0.95 \
  --min-moves 4 \
  --server-url $URL \
  --time 5.0 \
