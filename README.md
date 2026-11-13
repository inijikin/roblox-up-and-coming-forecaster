# Roblox Up-and-Coming top-10 forecaster

Forecasts Roblox's Up-and-Coming chart place by place for the make.fun bounty
`roblox-up-and-coming-forecast-01`: for each of the chart's top 10 places, a
probability for each game named being at that place, plus the top 10 in order.

Method: empirical rank-transition kernel. The chart is sorted by user growth,
not by player count, so the forecast models the rank itself: each game's
future rank is a normal distribution centred on its current rank plus its
rank band's drift, with a spread growing with the horizon (calibrated from the
history's own consecutive-read rank changes). A per-place newcomer reserve is
left unnamed and grows with the horizon, for games born after the history
ends. Standard library only, no network, deterministic.

Run locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
