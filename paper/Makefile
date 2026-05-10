# ter paper -- LaTeX build (requires mactex / texlive)
#
# Usage:
#   make           # build PDF
#   make clean     # remove build artifacts
#   make publish   # copy PDF to ~/naranjositos_page/downloads/

PAPER = ter_paper
PUBLISH_DIR = $(HOME)/naranjositos_page/downloads

$(PAPER).pdf: $(PAPER).tex references.bib arxiv.sty
	pdflatex $(PAPER).tex
	bibtex $(PAPER)
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc

distclean: clean
	rm -f $(PAPER).pdf

publish: $(PAPER).pdf
	mkdir -p $(PUBLISH_DIR)
	cp $(PAPER).pdf $(PUBLISH_DIR)/ter_paper.pdf
	@echo "Published $(PAPER).pdf -> $(PUBLISH_DIR)/ter_paper.pdf"

.PHONY: clean distclean publish
