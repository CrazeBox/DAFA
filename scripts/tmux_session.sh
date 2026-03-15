#!/bin/bash
# ==============================================================================
# DAFA联邦学习 - tmux会话管理脚本
# ==============================================================================
# 使用方法:
#   ./tmux_session.sh start     # 启动tmux会话
#   ./tmux_session.sh attach    # 连接到会话
#   ./tmux_session.sh stop      # 停止会话
#   ./tmux_session.sh status    # 查看状态
# ==============================================================================

SESSION_NAME="dafa"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

start_session() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "会话 '$SESSION_NAME' 已存在"
        echo "使用 './tmux_session.sh attach' 连接"
        exit 0
    fi
    
    echo "创建tmux会话: $SESSION_NAME"
    
    tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"
    
    # 窗口0: 主实验
    tmux send-keys -t $SESSION_NAME "source $VENV_DIR/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME "clear" C-m
    tmux send-keys -t $SESSION_NAME "echo '=== DAFA实验环境 ==='" C-m
    tmux send-keys -t $SESSION_NAME "echo '运行实验: python scripts/run_all_experiments.py --experiment all'" C-m
    
    # 窗口1: GPU监控
    tmux new-window -t $SESSION_NAME -n "gpu"
    tmux send-keys -t $SESSION_NAME:1 "watch -n 1 nvidia-smi" C-m
    
    # 窗口2: 日志监控
    tmux new-window -t $SESSION_NAME -n "logs"
    tmux send-keys -t $SESSION_NAME:2 "cd $PROJECT_DIR/logs" C-m
    tmux send-keys -t $SESSION_NAME:2 "ls -lt | head" C-m
    
    # 窗口3: 系统监控
    tmux new-window -t $SESSION_NAME -n "monitor"
    tmux send-keys -t $SESSION_NAME:3 "htop" C-m
    
    echo ""
    echo "tmux会话已创建!"
    echo ""
    echo "快捷键:"
    echo "  Ctrl+B 0    - 主实验窗口"
    echo "  Ctrl+B 1    - GPU监控窗口"
    echo "  Ctrl+B 2    - 日志窗口"
    echo "  Ctrl+B 3    - 系统监控窗口"
    echo "  Ctrl+B D    - 分离会话"
    echo ""
    echo "连接会话: ./tmux_session.sh attach"
}

attach_session() {
    if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "会话 '$SESSION_NAME' 不存在"
        echo "使用 './tmux_session.sh start' 创建"
        exit 1
    fi
    
    tmux attach-session -t $SESSION_NAME
}

stop_session() {
    if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "会话 '$SESSION_NAME' 不存在"
        exit 0
    fi
    
    echo "停止会话: $SESSION_NAME"
    tmux kill-session -t $SESSION_NAME
    echo "会话已停止"
}

show_status() {
    echo "=== tmux会话状态 ==="
    tmux list-sessions 2>/dev/null || echo "没有运行的会话"
    echo ""
    
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "会话 '$SESSION_NAME' 窗口:"
        tmux list-windows -t $SESSION_NAME
    fi
}

case "$1" in
    start)
        start_session
        ;;
    attach|a)
        attach_session
        ;;
    stop|kill)
        stop_session
        ;;
    status|s)
        show_status
        ;;
    *)
        echo "使用方法: $0 {start|attach|stop|status}"
        echo ""
        echo "命令:"
        echo "  start   - 创建并启动tmux会话"
        echo "  attach  - 连接到现有会话"
        echo "  stop    - 停止并删除会话"
        echo "  status  - 显示会话状态"
        exit 1
        ;;
esac
