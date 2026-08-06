#!/bin/bash
# 杀死指定进程及其所有子进程
# Usage: bash kill.sh <PID>

if [ -z "$1" ]; then
    echo "Usage: bash kill.sh <PID>"
    echo "  杀死指定进程及其所有子进程"
    exit 1
fi

PID=$1

# 检查进程是否存在
if ! kill -0 "$PID" 2>/dev/null; then
    echo "进程 $PID 不存在"
    exit 1
fi

# 递归收集所有子进程（包括子进程的子进程）
collect_children() {
    local parent=$1
    local children
    children=$(ps -o pid= --ppid "$parent" 2>/dev/null | tr -d ' ')
    for child in $children; do
        collect_children "$child"
    done
    echo "$parent"
}

# 收集进程树（子进程在前，父进程在后，保证先杀子再杀父）
PROCESS_TREE=$(collect_children "$PID")
TOTAL=$(echo "$PROCESS_TREE" | wc -l | tr -d ' ')

echo "进程树（共 $TOTAL 个进程）:"
while read -r p; do
    cmd=$(ps -o args= -p "$p" 2>/dev/null | head -c 80)
    printf "  PID=%-8s %s\n" "$p" "$cmd"
done <<< "$PROCESS_TREE"

# 先 SIGTERM 优雅终止
echo ""
echo "发送 SIGTERM..."
while read -r p; do
    kill "$p" 2>/dev/null
done <<< "$PROCESS_TREE"

# 等待最多 5 秒
for i in $(seq 1 5); do
    alive=false
    while read -r p; do
        if kill -0 "$p" 2>/dev/null; then
            alive=true
        fi
    done <<< "$PROCESS_TREE"
    if [ "$alive" = false ]; then
        break
    fi
    sleep 1
done

# 检查是否还有残留，强制 SIGKILL
remaining=false
while read -r p; do
    if kill -0 "$p" 2>/dev/null; then
        remaining=true
    fi
done <<< "$PROCESS_TREE"

if [ "$remaining" = true ]; then
    echo "部分进程未响应，发送 SIGKILL..."
    while read -r p; do
        kill -9 "$p" 2>/dev/null
    done <<< "$PROCESS_TREE"
fi

echo "已终止进程树（根 PID=$PID，共 $TOTAL 个进程）"
