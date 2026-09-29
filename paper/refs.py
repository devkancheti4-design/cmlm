"""PeerJ-style references, alphabetical; arXiv entries built from the arXiv API record."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "arxiv_meta.json")))
SPECIAL = {"Aaron van den Oord": "Oord A van den", "Swamynathan V P": "V P S"}


def author(name):
    if name in SPECIAL:
        return SPECIAL[name]
    parts = name.replace(".", " ").split()
    fam, given = parts[-1], parts[:-1]
    ini = "".join("-".join(g[0] for g in p.split("-")) if "-" in p else p[0] for p in given)
    return f"{fam} {ini}"


def arx(aid):
    m = META[aid]
    return ", ".join(author(a) for a in m["authors"]), m["published"][:4], m["title"]


def A(aid, year=None, venue=None):
    au, y, title = arx(aid)
    y = year or y
    tail = f" {venue}." if venue else " arXiv preprint."
    return f"{au}. {y}. {title}.{tail} Available at https://arxiv.org/abs/{aid}."


REFS = [
    ("Alayrac", A("2204.14198", venue="In: Advances in Neural Information Processing Systems 35 (NeurIPS 2022)")),
    ("Behrouz", A("2501.00663")),
    ("Chevalier", "Chevalier A, Wettig A, Ajith A, Chen D. 2023. Adapting language models to compress contexts. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 3829–3846. DOI: 10.18653/v1/2023.emnlp-main.232."),
    ("Dawid", "Dawid AP. 1984. Present position and potential developments: some personal views. Statistical theory: the prequential approach. Journal of the Royal Statistical Society, Series A (General) 147(2):278–292. DOI: 10.2307/2981683."),
    ("Deng", A("2412.17483")),
    ("Hardt", A("2305.18466")),
    ("He", A("2402.13449")),
    ("Huang", "Huang PS, He X, Gao J, Deng L, Acero A, Heck L. 2013. Learning deep structured semantic models for web search using clickthrough data. In: Proceedings of the 22nd ACM International Conference on Information and Knowledge Management (CIKM 2013). 2333–2338. DOI: 10.1145/2505515.2505665."),
    ("Kancheti", "Kancheti D. 2026. living-fused: a small deterministic memory for local language models. Available at https://github.com/devkancheti4-design/living-fused (accessed 28 September 2026)."),
    ("Khandelwal", A("1911.00172", year="2020", venue="In: 8th International Conference on Learning Representations (ICLR 2020)")),
    ("Lester", "Lester B, Al-Rfou R, Constant N. 2021. The power of scale for parameter-efficient prompt tuning. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3045–3059. DOI: 10.18653/v1/2021.emnlp-main.243."),
    ("Lewis", A("2005.11401", venue="In: Advances in Neural Information Processing Systems 33 (NeurIPS 2020)")),
    ("Li A", A("2606.09659")),
    ("Li XL", "Li XL, Liang P. 2021. Prefix-tuning: optimizing continuous prompts for generation. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 4582–4597. DOI: 10.18653/v1/2021.acl-long.353."),
    ("Liu F", A("2511.15244")),
    ("Liu H", A("2304.08485", venue="In: Advances in Neural Information Processing Systems 36 (NeurIPS 2023)")),
    ("McCloskey", "McCloskey M, Cohen NJ. 1989. Catastrophic interference in connectionist networks: the sequential learning problem. Psychology of Learning and Motivation 24:109–165. DOI: 10.1016/S0079-7421(08)60536-8."),
    ("Mu", A("2304.08467", venue="In: Advances in Neural Information Processing Systems 36 (NeurIPS 2023)")),
    ("Nijkamp", A("2609.19519")),
    ("Oord", A("1807.03748")),
    ("Packer", A("2310.08560")),
    ("Rebuffi", "Rebuffi SA, Kolesnikov A, Sperl G, Lampert CH. 2017. iCaRL: incremental classifier and representation learning. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 5533–5542. DOI: 10.1109/CVPR.2017.587."),
    ("Schmidhuber", "Schmidhuber J. 1992. Learning to control fast-weight memories: an alternative to dynamic recurrent networks. Neural Computation 4(1):131–139. DOI: 10.1162/neco.1992.4.1.131."),
    ("Sun", A("2507.22925")),
    ("V P", A("2603.06642")),
    ("Wang", A("2608.01672")),
    ("Weinberger", "Weinberger K, Dasgupta A, Langford J, Smola A, Attenberg J. 2009. Feature hashing for large scale multitask learning. In: Proceedings of the 26th Annual International Conference on Machine Learning (ICML 2009). 1113–1120. DOI: 10.1145/1553374.1553516."),
    ("Weston", A("1410.3916", year="2015", venue="In: 3rd International Conference on Learning Representations (ICLR 2015)")),
    ("Yu", A("2601.01885")),
    ("Zhang J", A("2604.07798").replace(". 2026. ", ". 2026a. ", 1)),
    ("Zhang Z", A("2608.00814").replace(". 2026. ", ". 2026b. ", 1)),
    ("Zhao", A("2601.00671")),
]

# In-text forms that must each match one reference (PeerJ: all cited, all listed)
CITES = {
    "Alayrac et al., 2022": "Alayrac", "Behrouz, Zhong & Mirrokni, 2024": "Behrouz", "Chevalier et al., 2023": "Chevalier",
    "Dawid, 1984": "Dawid", "Deng et al., 2024": "Deng", "Hardt & Sun, 2023": "Hardt", "He et al., 2024": "He",
    "Huang et al., 2013": "Huang", "Kancheti, 2026": "Kancheti", "Khandelwal et al., 2020": "Khandelwal",
    "Lester, Al-Rfou & Constant, 2021": "Lester", "Lewis et al., 2020": "Lewis", "Li et al., 2026": "Li A",
    "Li & Liang, 2021": "Li XL", "Liu & Qiu, 2025": "Liu F", "Liu et al., 2023": "Liu H", "McCloskey & Cohen, 1989": "McCloskey",
    "Mu, Li & Goodman, 2023": "Mu", "Nijkamp et al., 2026": "Nijkamp", "Oord, Li & Vinyals, 2018": "Oord",
    "Packer et al., 2023": "Packer", "Rebuffi et al., 2017": "Rebuffi", "Schmidhuber, 1992": "Schmidhuber",
    "Sun & Zeng, 2025": "Sun", "V P, 2026": "V P", "Wang et al., 2026": "Wang", "Weinberger et al., 2009": "Weinberger",
    "Weston, Chopra & Bordes, 2015": "Weston", "Yu et al., 2026": "Yu", "Zhang et al., 2026a": "Zhang J",
    "Zhang et al., 2026b": "Zhang Z", "Zhao & Jones, 2026": "Zhao",
}

if __name__ == "__main__":
    for k, r in REFS:
        print(r)
