#!/bin/bash

# 推送 Uni-LoRA MoE 分支到远程仓库的脚本

cd "$(dirname "$0")"

echo "=========================================="
echo "推送 Uni-LoRA MoE 分支到远程仓库"
echo "=========================================="

# 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "unilora_moe" ]; then
    echo "警告: 当前不在 unilora_moe 分支，切换到该分支..."
    git checkout unilora_moe
fi

# 显示远程仓库
echo ""
echo "当前远程仓库配置:"
git remote -v

echo ""
echo "请选择操作:"
echo "1. 推送到当前 origin (https://github.com/KaiyangLi1992/Uni-LoRA.git)"
echo "2. 添加新的远程仓库（您的 fork）并推送"
echo "3. 仅显示推送命令，手动执行"
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "推送到 origin/unilora_moe..."
        echo "注意: 如果遇到身份验证问题，请使用选项 2 添加您自己的远程仓库"
        git push -u origin unilora_moe
        ;;
    2)
        read -p "请输入您的 GitHub 用户名: " username
        read -p "请输入您的仓库名 (默认: Uni-LoRA): " reponame
        reponame=${reponame:-Uni-LoRA}
        
        echo ""
        echo "添加远程仓库: https://github.com/${username}/${reponame}.git"
        git remote add myfork "https://github.com/${username}/${reponame}.git" 2>/dev/null || \
        git remote set-url myfork "https://github.com/${username}/${reponame}.git"
        
        echo ""
        echo "推送到 myfork/unilora_moe..."
        git push -u myfork unilora_moe
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "手动推送命令:"
        echo "=========================================="
        echo ""
        echo "如果推送到当前 origin:"
        echo "  git push -u origin unilora_moe"
        echo ""
        echo "如果添加您自己的远程仓库:"
        echo "  git remote add myfork https://github.com/YOUR_USERNAME/YOUR_REPO.git"
        echo "  git push -u myfork unilora_moe"
        echo ""
        echo "如果使用 SSH (推荐):"
        echo "  git remote add myfork git@github.com:YOUR_USERNAME/YOUR_REPO.git"
        echo "  git push -u myfork unilora_moe"
        echo ""
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成！"

