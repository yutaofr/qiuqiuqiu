# Weekly Digest launchd Setup

## Overview

Use `launchd` on macOS to schedule the weekly digest after Friday market close. The repository ships a template only. Operators must fill in absolute paths before installation.

## Template

- [launchd/com.qiuqiuqiu.weekly.plist.template](/Users/weizhang/w/cycle-monitor-workspace/qiuqiuqiu/launchd/com.qiuqiuqiu.weekly.plist.template)

The template does not contain secrets or a `--week-end` placeholder.

## Install

1. Replace `/ABSOLUTE/PATH/TO/qiuqiuqiu` with the real checkout path.
2. Write the rendered plist to `~/Library/LaunchAgents/com.qiuqiuqiu.weekly.plist`.
3. Load it with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.qiuqiuqiu.weekly.plist`.
4. Optionally run `launchctl enable gui/$(id -u)/com.qiuqiuqiu.weekly`.

## Schedule

The template uses local time. Set the machine timezone so Friday 16:15 ET maps to the desired local time, or convert that schedule explicitly when editing the plist.

## Logs

The template writes stdout and stderr to:

- `logs/launchd/weekly.out.log`
- `logs/launchd/weekly.err.log`

## Removal

Unload the agent with `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.qiuqiuqiu.weekly.plist` and then remove the plist file if you no longer want the schedule.

