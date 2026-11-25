#!/bin/bash
# Publish web/site (page + data) to GitHub Pages: the gh-pages branch of the (public) windcast
# repository, https://chensy618.github.io/windcast/. Each call replaces that branch by one commit
# with no parent (force push), so the forecast data never piles up in the history and never
# touches main.
#
#   bash web/deploy_pages.sh                       # by hand
#   export WINDCAST_DEPLOY='bash web/deploy_pages.sh'   # after every cycle (web/slurm/scrontab.txt)
#
# PAGES_REPO / PAGES_BRANCH: where the site goes; Pages serves that branch, root folder.
# The compute nodes reach the internet only through $https_proxy: there git goes over SSH on port
# 443 (ssh.github.com) through the proxy (web/proxy_connect.py); on the login nodes directly.
set -euo pipefail
REPO=/cluster/home/siyan/github/windcast
PAGES_REPO=${PAGES_REPO:-chensy618/windcast}
PAGES_BRANCH=${PAGES_BRANCH:-gh-pages}
KEY=${PAGES_KEY:-$HOME/.ssh/id_ed25519_chensy618_account}
PAGES_REMOTE="ssh://git@ssh.github.com:443/$PAGES_REPO.git"
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30"
if [ -n "${https_proxy:-}${http_proxy:-}" ]; then
  SSH="$SSH -o ProxyCommand='/usr/bin/python3 $REPO/web/proxy_connect.py %h %p'"
fi
export GIT_SSH_COMMAND="$SSH"
WORK=${PAGES_WORK:-/cluster/work/projects/nn8106k/siyan/windcast/pages}
exec 9>"$WORK.lock"; flock -w 600 9               # the hourly observation job and a cycle may overlap
mkdir -p "$WORK"
rsync -a --delete --exclude .git "$REPO/web/site/" "$WORK/"
touch "$WORK/.nojekyll"                            # serve the files as they are
cd "$WORK"
# the work repository is kept: the commit pushed last time is what the remote branch points to,
# so a push only sends the files that changed (a few MB after an observation update)
[ -d .git ] || git init -q -b "$PAGES_BRANCH"
git checkout -q --orphan next
git add -A
git -c user.name="windcast-bot" -c user.email="windcast-bot@users.noreply.github.com" \
    commit -q -m "Windcast forecast $(date -u +%Y-%m-%dT%H:%MZ)"
git branch -q -M next "$PAGES_BRANCH"
git push -q -f "$PAGES_REMOTE" "$PAGES_BRANCH"
git reflog expire --expire=now --all && git gc -q --prune=now        # drop the older commits locally
echo "published $(du -sh . | cut -f1) to $PAGES_REPO $PAGES_BRANCH"
