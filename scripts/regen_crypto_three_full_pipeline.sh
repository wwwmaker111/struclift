#!/usr/bin/env bash
# Deprecated name: pipeline is now Botan + Crypto++ only (no BoringSSL).
# This script forwards to regen_botan_cryptopp_full_pipeline.sh.

exec "$(dirname "$0")/regen_botan_cryptopp_full_pipeline.sh" "$@"
