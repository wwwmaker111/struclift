#!/bin/bash
# 下载并编译 SQLite：O0 按 BB_Source_Alignment_Pipeline.md，O1/O2/O3 用于 Module A
# 产出: sqlite_o0..o3、sqlite_os、对应 .ll
# SQLite 用 amalgamation（单文件），不需要 configure，非常干净

set -e
SQLITE_YEAR="${SQLITE_YEAR:-2024}"
SQLITE_VERSION="${SQLITE_VERSION:-3450100}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/sqlite}"
DOWNLOAD_URL="https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip"
ZIPFILE="/tmp/sqlite-amalgamation-${SQLITE_VERSION}.zip"
DIR="${BASE_DIR}/sqlite-${SQLITE_VERSION}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

LIBS="-lpthread -ldl -lm"

mkdir -p "$BASE_DIR"

if [ ! -f "$ZIPFILE" ]; then
  echo "[1/5] 下载 sqlite-amalgamation-${SQLITE_VERSION} ..."
  wget -q -O "$ZIPFILE" "$DOWNLOAD_URL" || curl -sL -o "$ZIPFILE" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/5] 解压 ..."
  unzip -q -o "$ZIPFILE" -d "$BASE_DIR"
  mv "$BASE_DIR/sqlite-amalgamation-${SQLITE_VERSION}" "$DIR" 2>/dev/null || true
fi
cd "$DIR"

echo "[3/5] O0 (按 MD) + 产出 IR ..."
clang $MD_CFLAGS -DSQLITE_THREADSAFE=0 -c sqlite3.c -o sqlite3.o
clang $MD_CFLAGS -DSQLITE_THREADSAFE=0 -c shell.c -o shell.o
clang $MD_CFLAGS sqlite3.o shell.o -o sqlite_o0 $LIBS
rm -f sqlite3.o shell.o
echo "    -> $DIR/sqlite_o0"

echo "    [.ll] 生成 sqlite_o0.ll ..."
clang $MD_CFLAGS -DSQLITE_THREADSAFE=0 -emit-llvm -c sqlite3.c -o sqlite3.bc
clang $MD_CFLAGS -DSQLITE_THREADSAFE=0 -emit-llvm -c shell.c -o shell.bc
llvm-link sqlite3.bc shell.bc -o sqlite_o0.bc
llvm-dis sqlite_o0.bc -o sqlite_o0.ll
rm -f sqlite3.bc shell.bc sqlite_o0.bc
echo "    -> $DIR/sqlite_o0.ll"

echo "[4/5] O1 / O2 / O3 (ELF + IR) ..."
for opt in 1 2 3; do
  case $opt in 1) F="$O1_CFLAGS" ;; 2) F="$O2_CFLAGS" ;; 3) F="$O3_CFLAGS" ;; esac
  clang $F -DSQLITE_THREADSAFE=0 sqlite3.c shell.c -o "sqlite_o$opt" $LIBS
  echo "    -> $DIR/sqlite_o$opt"

  clang $F -DSQLITE_THREADSAFE=0 -emit-llvm -c sqlite3.c -o sqlite3.bc
  clang $F -DSQLITE_THREADSAFE=0 -emit-llvm -c shell.c -o shell.bc
  llvm-link sqlite3.bc shell.bc -o "sqlite_o${opt}.bc"
  llvm-dis "sqlite_o${opt}.bc" -o "sqlite_o${opt}.ll"
  rm -f sqlite3.bc shell.bc "sqlite_o${opt}.bc"
  echo "    -> $DIR/sqlite_o${opt}.ll"
done

echo "[5/5] 完成"
echo "  O0: $DIR/sqlite_o0  $DIR/sqlite_o0.ll"
echo "  O1/O2/O3: $DIR/sqlite_o1 $DIR/sqlite_o1.ll  $DIR/sqlite_o2 $DIR/sqlite_o2.ll  $DIR/sqlite_o3 $DIR/sqlite_o3.ll"
echo "  源码目录: $DIR (--src 指向此目录)"
