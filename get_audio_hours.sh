find -type f -name '*.wav' | xargs -I{} soxi -D {} | python -c "import sys; print( sum(float(l)/3600 for l in sys.stdin))"
