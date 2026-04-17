## Auto-Setup (paste this into Claude Desktop or web)

```
Set up Claude Code CLI for me on macOS. Follow these steps in order:

1. CHECK AND INSTALL NODE, NPM AND BUN
   Check Node: node --version
   - If missing or below v18, install via the official nvm installer:
       curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
     Reload terminal: source ~/.zshrc
     Then install the latest LTS:
       nvm install --lts
     Reload terminal: source ~/.zshrc
   Check npm: npm --version
   - npm comes bundled with Node — if Node is installed, npm should be present.
     If missing after Node install, reload terminal: source ~/.zshrc and retry.
   Check Bun: bun --version
   - If missing, install via the official Bun installer:
       curl -fsSL https://bun.sh/install | bash
     Reload terminal: source ~/.zshrc
   After installs, confirm all three: node --version && npm --version && bun --version

2. ENSURE PATH IS CORRECT
   Claude Code and non-interactive shells may not inherit your full PATH.
   Ensure your ~/.zshrc has these lines NEAR THE TOP (before any tool checks):

     # Ensure /usr/local/bin is on PATH (system tools)
     export PATH="/usr/local/bin:$PATH"

   If Homebrew is installed, also add:
     eval "$(/opt/homebrew/bin/brew shellenv)"

   nvm and Bun installers add their own PATH lines automatically.
   Verify they are present in ~/.zshrc, then reload: source ~/.zshrc

3. CHECK / INSTALL CLAUDE CODE CLI
   Run: which claude 2>/dev/null && claude --version
   - If a version number is returned: Claude Code is already installed.
     Skip the install below and proceed directly to step 4.
   - If NOT found: run:
       curl -fsSL https://claude.ai/install.sh | bash
     Reload terminal: source ~/.zshrc
     Then confirm: claude --version
   - Do NOT touch /Applications/Claude.app — that is the Desktop app, leave it alone.

4. CREATE OR UPDATE CLAUDE SETTINGS
   Run: mkdir -p ~/.claude
   If ~/.claude/settings.json already exists, merge the keys below into it,
   preserving any existing values not listed here.
   Otherwise, create it with exactly this content:
   {
     "env": {
       "ENABLE_LSP_TOOL": "1",
       "CLAUDE_CODE_NO_FLICKER": "1"
     },
     "statusLine": {
       "type": "command",
       "command": "bunx ccstatusline@latest",
       "padding": 0
     },
     "enabledPlugins": {
       "superpowers@claude-plugins-official": true,
       "frontend-design@claude-plugins-official": true,
       "claude-code-setup@claude-plugins-official": true,
       "frontend-design@claude-code-plugins": true,
       "typescript-lsp@claude-plugins-official": true,
       "ui-ux-pro-max@ui-ux-pro-max-skill": true
     },
     "extraKnownMarketplaces": {
       "claude-code-plugins": {
         "source": {
           "source": "github",
           "repo": "anthropics/claude-code"
         }
       },
       "ui-ux-pro-max-skill": {
         "source": {
           "source": "github",
           "repo": "nextlevelbuilder/ui-ux-pro-max-skill"
         }
       }
     },
     "effortLevel": "high",
     "autoUpdatesChannel": "latest",
     "skipDangerousModePermissionPrompt": true,
     "model": "sonnet",
     "permissions": {
       "defaultMode": "bypassPermissions"
     }
   }

5. INSTALL PLUGIN
   Run: claude plugin install frontend-design@claude-code-plugins

6. INSTALL GITHUB SKILLS
   Run each line:
   npx skills add cyxzdev/Uncodixfy@uncodixfy -g -y
   npx skills add vercel-labs/skills@find-skills -g -y
   npx skills add obra/superpowers@using-superpowers -g -y
   npx skills add microsoft/playwright-cli@playwright-cli -g -y
   npx skills add mattpocock/skills@write-a-skill -g -y
   npx skills add mattpocock/skills@improve-codebase-architecture -g -y
   npx skills add mattpocock/skills@tdd -g -y
   npx skills add mattpocock/skills@design-an-interface -g -y
   npx skills add mattpocock/skills@grill-me -g -y
   npx skills add mattpocock/skills@prd-to-plan -g -y
   npx skills add mattpocock/skills@write-a-prd -g -y
   npx skills add pbakaus/impeccable -g -y

7. INSTALL CUSTOM SKILLS
   Run: mkdir -p ~/.claude/skills
   Find custom-skills.zip — check ~/Downloads/custom-skills.zip first, then ~/Desktop/custom-skills.zip.
   Extract all three skill folders:
     unzip -o <path-to-zip> "dark-sequence-design-system/*" -d ~/.claude/skills/
     unzip -o <path-to-zip> "e2e-functional-audit/*" -d ~/.claude/skills/
     unzip -o <path-to-zip> "e2e-functional-fix/*" -d ~/.claude/skills/
     unzip -o <path-to-zip> "write-api-reference/*" -d ~/.claude/skills/
   Also extract the commands and scripts:
     mkdir -p ~/.claude/commands ~/.claude/scripts
     unzip -o <path-to-zip> "commands/review-my-usage.md" -d ~/.claude/
     unzip -o <path-to-zip> "scripts/token_usage_analyzer.py" -d ~/.claude/
   Confirm these exist:
     ~/.claude/skills/dark-sequence-design-system
     ~/.claude/skills/e2e-functional-audit
     ~/.claude/skills/e2e-functional-fix
     ~/.claude/skills/write-api-reference
     ~/.claude/commands/review-my-usage.md
     ~/.claude/scripts/token_usage_analyzer.py

8. INSTALL LANGUAGE SERVERS AND PLAYWRIGHT-CLI
   Run: npm install -g typescript-language-server typescript vscode-langservers-extracted bash-language-server intelephense pyright
   This covers: TypeScript, JavaScript, HTML, CSS, JSON, PHP, Python, Bash.
   Confirm: typescript-language-server --version && pyright --version
   Then install playwright-cli via bun (creates a proper binary Claude Code can spawn):
   Run: bun install -g playwright-cli
   Confirm: playwright-cli --version

9. CREATE GLOBAL CLAUDE.MD
   Run: mkdir -p ~/.claude
   Write ~/.claude/CLAUDE.md with exactly this content:
   # Claude Code — Global Rules

   ## Tool Priority: LSP FIRST
   **When navigating code, ALWAYS use LSP before grep, glob, bash, or Read-and-scan.** LSP is faster, more accurate, and cheaper on tokens. Only fall back to grep/glob for non-code text search or regex patterns. This is a hard rule, not a suggestion.

10. SET UP CCSTATUSLINE
   Create the directory: mkdir -p ~/.config/ccstatusline
   Write ~/.config/ccstatusline/settings.json with exactly this content:
   {
     "version": 3,
     "lines": [
       [
         { "id": "git1", "type": "git-branch", "color": "brightWhite", "backgroundColor": "hex:223d1e" },
         { "id": "3", "type": "session-clock", "color": "hex:282A36", "backgroundColor": "hex:FED500" },
         { "id": "5", "type": "reset-timer", "color": "hex:282A36", "backgroundColor": "hex:5388DE" },
         { "id": "7", "type": "session-usage", "color": "hex:282A36", "backgroundColor": "hex:7BB183" },
         { "id": "c0b00e70-cb48-4004-897d-0b0696859111", "type": "weekly-usage", "color": "hex:282A36", "backgroundColor": "hex:af82e5" }
       ],
       [
         { "id": "33c105a7-451a-46df-9149-a45721153a91", "type": "model", "color": "brightWhite", "backgroundColor": "hex:FF6D00" },
         { "id": "4961508e-84c8-4caa-9358-67a57b31d9ba", "type": "tokens-input", "color": "hex:282A36", "backgroundColor": "hex:82afc2" },
         { "id": "4b6f4aaa-b6bc-43d4-ac47-fe5d4eac6634", "type": "tokens-output", "color": "hex:282A36", "backgroundColor": "hex:9984ad" },
         { "id": "7e50b7f4-e593-4d8e-8b14-08422f84e518", "type": "tokens-total", "color": "brightWhite", "backgroundColor": "hex:fa5757" },
         { "id": "2d4bd415-32bf-482a-8fa3-086801ff2eec", "type": "context-percentage", "color": "hex:282A36", "backgroundColor": "hex:ffa83d" }
       ],
       [
         { "id": "73c572d8-8203-4c4e-8efd-f126d58881b6", "type": "session-name" }
       ]
     ],
     "defaultPadding": " ",
     "flexMode": "full-minus-40",
     "compactThreshold": 60,
     "colorLevel": 3,
     "inheritSeparatorColors": false,
     "globalBold": true,
     "powerline": {
       "enabled": true,
       "separators": ["\uE0B0"],
       "separatorInvertBackground": [false],
       "startCaps": [],
       "endCaps": ["\uE0B4"],
       "autoAlign": false
     }
   }

11. SET ENABLE_LSP_TOOL FOR ALL APPS (including Claude Desktop)
    The settings.json env block only affects Claude Code CLI. To cover Claude Desktop
    and every other process, export the variable from your shell profile.

    Detect your shell: echo $SHELL
    - If /bin/zsh:  add to ~/.zshrc
    - If /bin/bash: add to ~/.bash_profile

    Run:
      echo 'export ENABLE_LSP_TOOL=1' >> ~/.zshrc
    For bash:
      echo 'export ENABLE_LSP_TOOL=1' >> ~/.bash_profile

    Reload: source ~/.zshrc  (or source ~/.bash_profile)
    Then restart Claude Desktop so it inherits the new env var.

12. ADD CLAUDE ALIAS (skip permissions)
    This makes typing "claude" in any terminal automatically run with --dangerously-skip-permissions.

    Run (zsh):
      echo 'claude() { command claude --dangerously-skip-permissions "$@"; }' >> ~/.zshrc
    For bash:
      echo 'claude() { command claude --dangerously-skip-permissions "$@"; }' >> ~/.bash_profile

    Reload: source ~/.zshrc  (or source ~/.bash_profile)

    For VS Code integrated terminal: ensure VS Code uses zsh or bash as the default shell,
    so it sources the profile above automatically on launch.

    Confirm: open a new terminal and run: claude --version
    (It should launch without the permissions prompt.)

13. INSTALL JETBRAINS MONO NERD FONT
    Download and install from the official Nerd Fonts release:
      curl -fsSL -o /tmp/JetBrainsMono.zip "https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.zip"
      mkdir -p /tmp/JetBrainsMono && unzip -o /tmp/JetBrainsMono.zip -d /tmp/JetBrainsMono
      cp /tmp/JetBrainsMono/*.ttf ~/Library/Fonts/
      rm -rf /tmp/JetBrainsMono /tmp/JetBrainsMono.zip
    After install, restart your terminal before proceeding.

14. CONFIGURE VSCODE TERMINAL FONT
    Open VS Code settings.json (Cmd+Shift+P → "Open User Settings JSON") and add or update:
      "terminal.integrated.fontFamily": "JetBrainsMono Nerd Font",
      "terminal.integrated.minimumContrastRatio": 1
    Save the file. Restart VS Code or reload the window (Cmd+Shift+P → "Developer: Reload Window").

15. VERIFY
    Confirm all config files exist and are valid JSON.
    Tell me to open a new terminal and run: claude
    Inside Claude, /skills should list 14+ skills and /plugins should show frontend-design enabled.
    The status line should render with Powerline separators at the bottom of the terminal.
```
