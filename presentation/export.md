 To re-export anytime (one command from the project root):
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless=new --disable-gpu --no-sandbox \
    --print-to-pdf="presentation/team_10_Presentation.pdf" \
    --print-to-pdf-no-header --no-pdf-header-footer \
    --paper-width=20 --paper-height=11.25 \
    --margin-top=0 --margin-bottom=0 --margin-left=0 --margin-right=0 \
    "file://$(pwd)/presentation/team_10_Presentation.html"