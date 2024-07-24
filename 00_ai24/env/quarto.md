# quarto


## from qmd to html

```
### install the tool
https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.55/quarto-1.5.55-linux-amd64.deb
sudo dpkg -i quarto-1.5.55-linux-amd64.deb
### Extensions Install the following:
quarto add quarto-ext/attribution
quarto add EmilHvitfeldt/quarto-roughnotation
quarto add mcanouil/quarto-iconify
quarto add fradav/quarto-revealjs-animate
### convert from repo to html:
quarto render ImageSegmentation.qmd --to revealjs
```