#!/bin/bash
 
# 获取脚本所在目录
DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
# 使用 autopep8 格式化指定目录中的所有 .py 文件
autopep8 --max-line-length 79  --recursive --in-place --aggressive --aggressive --verbose .
 
# 提示完成格式化
echo "Formatting done!"
