#!/bin/bash
DEST_DIR="large_files_for_huggingface"
find . -type f -size +10M -not -path "./${DEST_DIR}/*" | while read -r file; do
  mkdir -p "${DEST_DIR}/$(dirname "${file}")"
  mv "${file}" "${DEST_DIR}/${file}"
  echo "已移动: ${file}"
done
echo "所有大于10MB的文件已移动到 ${DEST_DIR} 目录。" 