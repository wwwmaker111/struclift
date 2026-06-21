#!/usr/bin/env bash
# 仅在已有 sqlite amalgamation 目录中编译 -Os，产出 sqlite_os、sqlite_os.ll
# 用法: bash scripts/os_only/compile_os_sqlite.sh <sqlite目录>
set -euo pipefail
DIR="${1:?用法: $0 <sqlite amalgamation 目录>}"
[ -f "$DIR/sqlite3.c" ] && [ -f "$DIR/shell.c" ] || { echo "需要 sqlite3.c/shell.c: $DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
LIBS="-lpthread -ldl -lm"

cd "$DIR"
clang $OS_CFLAGS -DSQLITE_THREADSAFE=0 sqlite3.c shell.c -o sqlite_os $LIBS
echo "    -> $DIR/sqlite_os"
clang $OS_CFLAGS -DSQLITE_THREADSAFE=0 -emit-llvm -c sqlite3.c -o sqlite3.bc
clang $OS_CFLAGS -DSQLITE_THREADSAFE=0 -emit-llvm -c shell.c -o shell.bc
llvm-link sqlite3.bc shell.bc -o sqlite_os.bc
llvm-dis sqlite_os.bc -o sqlite_os.ll
rm -f sqlite3.bc shell.bc sqlite_os.bc
echo "    -> $DIR/sqlite_os.ll"
echo "完成 Os: sqlite_os, sqlite_os.ll"
