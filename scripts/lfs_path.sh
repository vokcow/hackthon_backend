# Source from bash/zsh before git push/pull when git-lfs is not found:
#   source scripts/lfs_path.sh
#
# Homebrew on Apple Silicon installs git-lfs under /opt/homebrew/bin, which is
# often missing from PATH in Cursor, conda, or minimal shells.

if [[ -x /opt/homebrew/bin/git-lfs ]]; then
  export PATH="/opt/homebrew/bin:${PATH}"
elif [[ -x /usr/local/bin/git-lfs ]]; then
  export PATH="/usr/local/bin:${PATH}"
fi
