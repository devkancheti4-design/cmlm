"""Ready-to-paste answers for the PeerJ submission form, and the checklist of what only the author can do."""
import os
from docxgen import Doc
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission")
d = Doc(title="Submission form answers", author="Devieswar Kancheti", line_numbers=False, line=276)
d.title_p("PeerJ Computer Science: submission form answers")
d.p("Paste each block into the matching field of the online form. Items in square brackets are for you to fill in.")

d.heading("Article type and subject areas", 2)
d.p("Article type: Research Article. The manuscript also meets the AI Application criteria (motivation for the technique, "
    "data generation, metric justification, computing infrastructure, limitations).")
d.p("Suggested subject areas: Artificial Intelligence; Computational Linguistics; Data Mining and Machine Learning.")

d.heading("Funding statement", 2)
d.p("The author received no funding for this work.")

d.heading("Competing interests", 2)
d.p("The author is the developer of living-fused, whose exact-key memory component is used and evaluated in this work. "
    "The author declares no other competing interests.")

d.heading("Author contributions", 2)
d.p("Devieswar Kancheti conceived and designed the method and the experiments, directed the analyses, prepared the "
    "figures and tables, authored and reviewed drafts of the article, and approved the final draft.")

d.heading("Data availability", 2)
d.p("The following information was supplied regarding data availability: the code that generates the synthetic world, "
    "runs every experiment and regenerates every table, the results used in the paper, and the per-turn sizes of the "
    "real session (without any message text) are available in the Supplemental Files (Supplemental Code S1 and "
    "Supplemental Data S1). [If you archive the repository on Zenodo, add its DOI here.]")

d.heading("Declaration on the use of generative AI", 2)
d.p("Claude (Anthropic), models Claude Fable 5.1 and Claude Opus 5.5, accessed through Claude Code version 2.1.92, was "
    "used under the author's direction to write and run the prototype code and experiments, to produce the figures, to "
    "draft and revise the manuscript text, and to check references against arXiv and Crossref. In one experiment the "
    "model acted as the frontier model. The author confirms the originality and accuracy of the content, has checked the "
    "terms of use of the tool and confirms their suitability for publication, and takes full responsibility for the "
    "integrity of the whole content, including the accuracy of the references. The statement also appears in the "
    "Acknowledgements.")

d.heading("Preprint", 2)
d.p("If you post the paper on arXiv first, give its identifier in the form. PeerJ accepts submissions that have appeared "
    "on preprint servers, including arXiv.")

d.heading("Files to upload", 2)
d.bullets(["Manuscript.docx, the manuscript with the author cover page first.",
           "Figure 1.png to Figure 6.png, one file per figure (vector PDFs are also supplied).",
           "Table 1.docx to Table 11.docx, one file per table.",
           "Supplemental Code S1.zip and Supplemental Data S1.zip.",
           "Titles and legends: copy them from the file of figure and table titles and legends into the upload form."])

d.heading("Only you can do these", 2)
d.bullets(["Fill in your department, institution, city, country, postal address and institutional email on the cover page. "
           "The author information you enter online must match the cover page exactly.",
           "Read the whole manuscript and confirm every claim. The generative AI declaration says you have done so.",
           "Decide on the licence: PeerJ publishes every article under a Creative Commons Attribution licence (CC BY). You "
           "keep the copyright, but anyone may reuse the article, including commercially, with attribution. This is "
           "different from the all-rights-reserved arXiv option you chose earlier.",
           "Optionally archive the code on Zenodo to obtain a DOI, and add it to the data availability statement.",
           "Pay or arrange the article processing charge; check the current fee and any institutional agreement on PeerJ's site."])
d.save(os.path.join(OUT, "Submission form answers.docx")); print("saved")
