# (C) 2023 Daniel Mulnaes
# All rights reserved.
# Please contact dmwp.kk at gmail.com if you have any questions about this program.
from dataclasses import dataclass
import pandas as pd
import sys
from tqdm import tqdm


@dataclass
class ProteinTag:
    name: str
    sequence: str
    origin: str
    method: str
    reference: str
    length: int

    def info(self):
        info_string = (
            f"Name:     {self.name}\n"
            f"Sequence: {self.sequence}\n"
            f"Length:   {self.length}\n"
            f"Origin:   {self.origin}\n"
            f"Method:   {self.method}\n"
            f"Ref:      {self.reference}\n"
        )
        print(info_string)
        print(f"-" * 70)


@dataclass
class ProteinTagHit(ProteinTag):
    start: int
    end: int
    query_aln: str
    tag_aln: str
    location: str
    identity: float
    conservation: float

    def info(self):
        info_string = (
            f"Name:     {self.name}\n"
            f"Sequence: {self.sequence}\n"
            f"Location: {self.start+1}-{self.end+1}\n"
            f"Identity: {self.identity}\n"
            f"Length:   {self.length}\n"
            f"Origin:   {self.origin}\n"
            f"Method:   {self.method}\n"
            f"Ref:      {self.reference}\n"
        )
        print(info_string)
        print(f"-" * 70)


class TagFinder:
    def __init__(self, identity_cutoff=90, margin_cutoff=3):
        """
        Class for finding tags in sequences
        :param identity_cutoff: Minimum sequence identity required between a tag and a query
        :param margin_cutoff: Maximum number of residues between tags and termini to classify
        tags as belonging to a termini
        """
        self.tags = []
        self.identity_cutoff = identity_cutoff
        self.margin_cutoff = margin_cutoff

        # Tags that mediate covalent binding to a protein
        # which can be used as bait for purification
        covalent_tags = [
            (
                "Isopep-Tag",
                "TDKDMTITFTNKKDAE",
                "Pilin-C covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/20235501/",
            ),
            (
                "Spy-Tag",
                "AHIVMVDAYKPTK",
                "SpyCatcher protein covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/22366317/",
            ),
            (
                "Snoop-Tag",
                "KLGDIEFIKVNK",
                "SnoopCatcher protein covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/26787909/",
            ),
            (
                "SnoopJr-Tag",
                "KLGSIEFIKVNK",
                "SnoopCatcher or DogCatcher (via SnoopLigase) protein covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/29402082/",
            ),
            (
                "Dog-Tag",
                "DIPATYEFTDGKHYITNEPIPPK",
                "DogCatcher protein covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/34324879/",
            ),
            (
                "Sdy-Tag",
                "DPIVMIDNDKPIT",
                "SdyCatcher protein covalent binding",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/27783674/",
            ),
        ]
        # Tags that mediate biotynilation which causes strong
        # binding to the streptavidin protein which can be used
        # as bait for purification
        biotinylation_tags = [
            (
                "AviTag",
                "GLNDIFEAQKIEWHE",
                "Binding to Streptavidin via biotinylation",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/17379573/",
            ),
            (
                "Strep-Tag",
                "WSHPQFEK",
                "Binding to Streptavidin via biotinylation",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/8636976/",
            ),
            (
                "Strep-Tag",
                "AWAHPQPGG",
                "Binding to Streptavidin via biotinylation",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/8636976/",
            ),
            (
                "SBP-Tag",
                "MDEKTTGWRGGHVVEGLAGELEQLRARLEHHPQGQREP",
                "Binding to Streptavidin via biotinylation",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/11722181/",
            ),
        ]
        # Tags that are recognized by antibodies which can be
        # used as bait for antibody affinity purification
        antibody_tags = [
            (
                "T7-Tag",
                "MASMTGGQQMG",
                "Antibody affinity purification",
                "Bacteriophage T7 major capsid protein",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "V5-Tag",
                "GKPIPNPLLGLDST",
                "Antibody affinity purification",
                "Bacteriophage V5 major capsid protein",
                "https://pubmed.ncbi.nlm.nih.gov/29104895/",
            ),
            (
                "V5-Tag",
                "IPNPLLGLD",
                "Antibody affinity purification",
                "Bacteriophage V5 major capsid protein",
                "https://pubmed.ncbi.nlm.nih.gov/29104895/",
            ),
            (
                "B-Tag",
                "QYPALT",
                "Antibody affinity purification",
                "Bluetongue virus",
                "https://pubmed.ncbi.nlm.nih.gov/6306033/",
            ),
            (
                "HA-Tag",
                "YPYDVPDYA",
                "Antibody affinity purification",
                "Human Influenza Virus Hemagglutinin",
                "https://pubmed.ncbi.nlm.nih.gov/16921383/",
            ),
            (
                "HSV-Tag",
                "QPELAPED",
                "Antibody affinity purification",
                "Herpes Simplex Virus type 1",
                "https://pubmed.ncbi.nlm.nih.gov/30502324/",
            ),
            (
                "S1-Tag",
                "NANNPDWDF",
                "Antibody affinity purification",
                "Hepatitis B virus preS1 protein",
                "https://pubmed.ncbi.nlm.nih.gov/14659901/",
            ),
            (
                "VSV-Tag",
                "YTDIEMNRLGK",
                "Antibody affinity purification",
                "Visucular Stomatitis Virus Envelope Glycoprotein",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Rho1D4-Tag",
                "TETSQVAPA",
                "Antibody affinity purification",
                "Bovine Rhodopsin",
                "https://pubmed.ncbi.nlm.nih.gov/24943310/",
            ),
            (
                "Myc-Tag",
                "CEQKLISEEDL or EQKLISEEDL",
                "Antibody affinity purification",
                "c-Myc",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Universal-Tag",
                "HTTPHH",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "E-Tag",
                "GAPVPYPDPLEPR",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/29021300/",
            ),
            (
                "E2-Tag",
                "SSTSSDFRDR",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "EE-Tag",
                "EYMPME",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "EE-Tag",
                "EFMPME",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "NE-Tag",
                "TKENPRSNQEESYDDNES",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24040167/",
            ),
            (
                "KT3-Tag",
                "KPPTPPPEPET",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/8746622/",
            ),
            (
                "ALFA-Tag",
                "SRLEEELRRRLTE",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/33673130/",
            ),
            (
                "FLAG-Tag",
                "DYKDDDDK",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/11694294/",
            ),
            (
                "Spot-Tag",
                "PDRVRAVSHWSS",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/29500346/",
            ),
            (
                "C-Tag",
                "EPEA",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/20620148/",
            ),
            (
                "Xpress-Tag",
                "DLYDDDDK",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/35542178/",
            ),
            (
                "ProteinC-Tag",
                "EDQVDPRLIDGK",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/19892176/",
            ),
            (
                "AU1-Tag",
                "DTYRYI",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/23015698/",
            ),
            (
                "AU5-Tag",
                "TDFYLK",
                "Antibody affinity purification",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/23015698/",
            ),
        ]

        # Tags that contain poly-amino acid repeats which
        # can be used for chelation or ion-exchange purification
        poly_tags = [
            (
                "PolyCys-Tag",
                "CCCC",
                "Reducing agent e.g. DTT or beta-ME",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/3142291/",
            ),
            (
                "PolyGlu-Tag",
                "EEEEEE",
                "Anion-exchange column",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24056174/",
            ),
            (
                "PolyPhe-Tag",
                "FFFFFFFFFFF",
                "Phenyl-Sepharose or Ethylene glycolexchange column",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/3142291/",
            ),
            (
                "GlyHis-Tag",
                "GHHHH",
                "Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/34520613/",
            ),
            (
                "GlyHis-Tag",
                "GHHHHHH",
                "Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/34520613/",
            ),
            (
                "GlyHis-Tag",
                "GSSHHHHHH",
                "Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/34520613/",
            ),
            (
                "HAT-Tag",
                "KDHLIHNVHKEFHAHAHNK",
                "Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
        ]

        for n in range(5, 10):
            repeat = (
                "PolyArg-Tag",
                "R" * n,
                "Cation-exchange column",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            )
            poly_tags.append(repeat)

        for n in range(5, 17):
            repeat = (
                "PolyAsp-Tag",
                "D" * n,
                "Cation-exchange column",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            )
            poly_tags.append(repeat)

        for n in range(2, 11):
            repeat = (
                "PolyHis-Tag",
                "H" * n,
                "Divalent Ion Chelate (Ni2+, Co2+, Cu2+, Zn2+)",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            )
            poly_tags.append(repeat)

        other_tags = [
            (
                "Calmodulin-Tag",
                "KRRWKKNFIAVSAANRFKKISSSGAL",
                "Calmodulin affinity purification",
                "Synthetic peptide",
                "",
            ),
            (
                "iCapTag",
                "MIKIATRKYLGKQNVYGIGVERDHNFALKNGFIAHN",
                "Resin Capture",
                "Intein from Nosdoc Punctiforme",
                "https://pubmed.ncbi.nlm.nih.gov/29516483/",
            ),
            (
                "S-Tag",
                "KETAAAKFERQHMDS",
                "Binding to S-fragment of RNase A",
                "Ribonuclease",
                "https://pubmed.ncbi.nlm.nih.gov/11036653/",
            ),
            (
                "SofTag 1",
                "SLAELLNAGLGGS",
                "",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24334194/",
            ),
            (
                "SofTag 3",
                "TQDPSRVG",
                "",
                "Synthetic peptide",
                "https://pubmed.ncbi.nlm.nih.gov/24334194/",
            ),
        ]

        # Fusion proteins used as tags for purification TODO not implemented yet
        protein_tags = [
            (
                "ABP-Fusion",
                "albumin-binding protein 137AA",
                "bait with albumin",
                "Protein fusion" "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "AP-fusion",
                "alkaline phosphatase 444AA",
                "bait with antibody",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "BCCP-fusion",
                "biotin-carboxy carrier protein 100AA",
                "bait with streptavidin+biotin",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "CAT-fusion",
                "Chloramphenicol Acetyl Transferase 218AA",
                "bait with Chloramphenicol",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Cellulose-BD-fusion",
                "Cellulose binding domain 27-189AA",
                "bait with cellulose (family 1 CBDs) or ethylene glycol (family 2 & 3)",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Chitin-BD-fusion",
                "Chitin binding domain 51AA",
                "bait with chitin",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Choline-BD-fusion",
                "Choline binding domain 145AA",
                "bait with Choline ",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "DHFR-fusion",
                "Dihydropholate reductase 227AA",
                "bait with methotrexate and folate ",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "GBP-fusion",
                "Galactose binding protein 509AA",
                "bait with galactose",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "MBP-fusion",
                "Maltose binding protein 396AA",
                "bait with maltose or amylose",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "GST-fusion",
                "Gluthathione S-transferase 211AA",
                "bait with glutathione",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "LacZ-fusion",
                "LacZ 1024AA",
                "bait with APTG in borate buffer",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Profinity eXact-fusion",
                "Profinity eXact protein 75AA",
                "bait with subtilisin protease",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "ProteinA-fusion",
                "Stephylococcal protein A 280AA",
                "Antibody",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "ProteinG-fusion",
                "Stephylococcal protein G 280AA",
                "Amylose",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Streptavidin-fusion",
                "Streptavidin 159AA",
                "Biotin",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "SUMO-fusion",
                "Small ubiquitin-like modifier 100AA",
                "ni-NTA, NI-IMAC bait",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "T7-fusion",
                "Bacteriophage T7 major capsid protein 260AA",
                "Antibody",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "TrpE-fusion",
                "TrpE protein 25-336AA",
                "Antibody",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "Thioredoxin-fusion",
                "Thioreduxin protein 109AA",
                "Reducing agent eg DDT, Beta-ME",
                "https://pubmed.ncbi.nlm.nih.gov/24510596/",
            ),
            (
                "HaloTag-fusion",
                "haloalkane dehalogenase enzyme DhaA XXXAA",
                "Covalent resin Binding",
                "https://pubmed.ncbi.nlm.nih.gov/19464373/",
            ),
            ("NusA-fusion", "", "", ""),
        ]

        default_tags = (
            biotinylation_tags + antibody_tags + covalent_tags + poly_tags + other_tags
        )
        for tag in default_tags:
            name, sequence, method, origin, reference = tag
            self.add_tag(
                name=name,
                sequence=sequence,
                method=method,
                origin=origin,
                reference=reference,
            )

    def add_tag(self, name: str, sequence: str, method: str, origin: str, reference: str):
        """
        A Function that adds a tag to this class
        :param name:
        :param sequence:
        :param method:
        :param origin:
        :param reference:
        :return:
        """
        self.tags.append(
            ProteinTag(
                name=name,
                sequence=sequence,
                method=method,
                origin=origin,
                reference=reference,
                length=len(sequence),
            )
        )

    def tag_search(self, tag: ProteinTag, query_seq: str) -> [ProteinTagHit]:
        """
        Use gap-less alignment to find tag hits of all
        possible tags
        :param tag: The tag to be searched for
        :param query_seq: The query sequence to search
        :return: A list of ProteinTagHit classes
        """
        query_len = len(query_seq)
        hits = []
        for i in range(query_len - tag.length + 1):
            query_sub = query_seq[i: i + tag.length]
            identity = round(
                [t == q for t, q in zip(tag.sequence, query_sub)].count(True)
                / tag.length
                * 100,
                2,
            )
            if identity >= self.identity_cutoff:
                tag_aln = "-" * i + tag.sequence + "-" * (query_len - i - tag.length)
                hit = ProteinTagHit(
                    name=tag.name,
                    sequence=tag.sequence,
                    method=tag.method,
                    origin=tag.origin,
                    reference=tag.reference,
                    length=tag.length,
                    start=i,
                    end=i + tag.length,
                    query_aln=query_seq,
                    tag_aln=tag_aln,
                    identity=identity,
                    location="Unknown",
                    conservation=0,
                )
                hits.append(hit)
        if len(hits) == 0:
            return None
        return hits

    @staticmethod
    def tag_hit_overlap(hit1: ProteinTagHit, hit2: ProteinTagHit) -> bool:
        """
        Test if two tag hits are overlapping.
        :param hit1:
        :param hit2:
        :return: True/False
        """
        overlap12 = (
            hit2.start <= hit1.start <= hit2.end and hit2.start <= hit1.end <= hit2.end
        )
        overlap21 = (
            hit1.start <= hit2.start <= hit1.end and hit1.start <= hit2.end <= hit1.end
        )
        return overlap12 or overlap21

    def get_tag_hits(self, sequence: str) -> [ProteinTagHit]:
        """
        Get non-redundant tags in an input sequence.
        Args:
            sequence:      Input sequence in which the tags are found.
        Returns:
            tag_hits:      Output non-redundant list of ProteinTagHits.
        """
        # Find all hits and group by tag name
        all_hits = {}
        for tag in self.tags:
            hits = self.tag_search(tag, sequence)
            if hits:
                for hit in hits:
                    if hit.name not in all_hits:
                        all_hits[hit.name] = [hit]
                    else:
                        all_hits[hit.name] += [hit]

        # Remove redundant hits that overlap
        non_redundant = []
        for name in all_hits:
            kept_hits = []
            for hit1 in sorted(all_hits[name], key=lambda x: x.length, reverse=True):
                overlap = False
                for hit2 in kept_hits:
                    if self.tag_hit_overlap(hit1, hit2):
                        overlap = True
                        break
                if not overlap:
                    kept_hits.append(hit1)
            non_redundant += kept_hits

        return non_redundant

    def get_termini(self, tag_hits: [ProteinTagHit], sequence: str) -> dict:
        """
        Find N-terminal and C-terminal regions containing tags.
        Args:
            tag_hits:      Input list of ProteinTagHits.
            sequence:      Input sequence in which the tags are found.
        Returns:
            termini:       Output dictionary with the N-terminal and
                           C-terminal regions containing tags.
        """
        # Find the maximum n-terminus cutoff and minimum c-terminus cutoff
        max_n_terminus = 0
        for h, hit in enumerate(sorted(tag_hits, key=lambda x: x.start)):
            if h == 0 and hit.start > self.margin_cutoff:
                break
            if max_n_terminus:
                if hit.start > max_n_terminus + self.margin_cutoff:
                    break
            max_n_terminus = hit.end
        min_c_terminus = len(sequence)
        for h, hit in enumerate(sorted(tag_hits, key=lambda x: x.end, reverse=True)):
            if h == 0 and len(hit.query_aln) - hit.end > self.margin_cutoff:
                break
            if min_c_terminus:
                if hit.end < min_c_terminus - self.margin_cutoff:
                    break
            min_c_terminus = hit.start

        termini = {"N-terminus": (0, max_n_terminus),
                   "C-terminus": (min_c_terminus, len(sequence)),
                   }
        return termini

    @staticmethod
    def assign_locations(tag_hits: [ProteinTagHit], termini: dict) -> [ProteinTagHit]:
        """
        Assigns tags a location flag specifying if they
        are in the N-terminus, C-terminus or internal.
        Args:
            tag_hits:      Input list of ProteinTagHits.
            termini:       Input dictionary with the N-terminal and
                           C-terminal regions containing tags.
        Returns:
            tag_hits:      Output list of ProteinTagHits with updated location attributes.
        """

        # Separate hits into n-terminal, c-terminal and internal hits
        max_n_terminus = termini["N-terminus"][-1]
        min_c_terminus = termini["C-terminus"][0]
        n_terminal_hits = sorted(
            [h for h in tag_hits if h.end <= max_n_terminus], key=lambda x: x.start
        )
        c_terminal_hits = sorted(
            [h for h in tag_hits if h.start >= min_c_terminus], key=lambda x: x.start
        )
        internal_hits = [
            hit
            for hit in tag_hits
            if hit not in n_terminal_hits and hit not in c_terminal_hits
        ]
        for tag in n_terminal_hits:
            tag.location = "N-terminus"
        for tag in c_terminal_hits:
            tag.location = "C-terminus"
        for tag in internal_hits:
            tag.location = "Internal"
        assigned_hits = sorted(
            n_terminal_hits + c_terminal_hits + internal_hits, key=lambda x: x.start
        )
        return assigned_hits

    @staticmethod
    def assign_conservation(tag_hits: [ProteinTagHit], query_aln: str, reference_aln: str) -> [ProteinTagHit]:
        """
        Assigns conservation to tags based on a pairwise
        alignment between the query sequence and a biological
        reference sequence.
        Calculate the conservation of the tags based on the
        alignment. Conserved tags (high identity) are likely
        to be false positives, since the sequence is conserved
        in the reference.
        Args:
            query_aln:     Input alignment of the query sequence.
            reference_aln: Input alignment of the reference sequence.
            tag_hits:      Input list of ProteinTagHits.
        Returns:
            tags:          Output list of ProteinTagHits with updated conservation attributes.
        Raises:
            TypeError:     If the input alignments do not have equal length
        """
        if len(query_aln) != len(reference_aln):
            raise TypeError(
                "Alignment strings query_aln and reference_aln must be equal length."
            )

        identity_mask = []
        for que, ref in zip(query_aln, reference_aln):
            if que == "-":
                continue
            elif ref == "-" or que != ref:
                identity_mask.append(0)
            else:
                identity_mask.append(1)

        for tag in tag_hits:
            reference_identity = identity_mask[tag.start: tag.end]
            tag.conservation = round(sum(reference_identity) / tag.length * 100, 2)

        return tag_hits

    def find_tags(
        self,
        sequence: str,
        query_aln: str = "",
        reference_aln: str = "",
        verbose: bool = True,
    ) -> ([ProteinTagHit], dict):
        """
        Find tags in an input sequence and return tags and termini
        Args:
            sequence:      Input sequence to find tags in.
            query_aln:     (Optional) Input alignment of the query sequence.
            reference_aln: (Optional) Input alignment of the reference sequence.
            verbose:       If True, print the tags that were found.
        Returns:
            tags, termini: Tuple containing a list of ProteinTagHits and a dictionary
                           with the N-terminal and C-terminal regions containing tags.
        Raises:
            TypeError:     If the query_aln and reference_aln do not have equal length.
            TypeError:     If the query_aln does not have the same sequence as sequence.
        Example:
            >>> tf = TagFinder()
            >>> my_tags, my_termini = tf.find_tags.(sequence='GHHHHHHMYNAMEISCARL')
            >>> print(my_tags[0].sequence)
            >>> 'GHHHHHH'
            >>> print(my_termini)
            >>> '{"N-terminus": (0, 7), "C-terminus": (18, 18)}'
        """
        tag_hits = self.get_tag_hits(sequence)

        # Validate tags if a query and reference alignment is provided
        if len(query_aln) != 0 and len(reference_aln) != 0:
            if len(query_aln) != len(reference_aln):
                raise TypeError(
                    "Error: The provided query and reference alignments must have equal length"
                )
            if query_aln.replace("-", "") != sequence:
                raise TypeError(
                    "Error: The provided query alignment does not match the input sequence"
                )

            valid_hits = self.assign_conservation(tag_hits=tag_hits,
                                                  query_aln=query_aln,
                                                  reference_aln=reference_aln)
            # Keep tags with low conservation, as these are not found in the reference sequence
            tag_hits = [
                tag
                for tag in valid_hits
                if tag.conservation <= 100 - self.identity_cutoff
            ]

        termini = self.get_termini(tag_hits, sequence)
        all_tags = self.assign_locations(tag_hits, termini)

        # Separate hits into n-terminal, c-terminal and internal hits
        n_terminal_tags = [tag for tag in all_tags if tag.location == "N-terminus"]
        c_terminal_tags = [tag for tag in all_tags if tag.location == "C-terminus"]
        internal_tags = [tag for tag in all_tags if tag.location == "Internal"]

        if verbose:
            if n_terminal_tags:
                print("=" * 70)
                print("N-terminus tags:")
                print("=" * 70)
                for tag in n_terminal_tags:
                    tag.info()
            if c_terminal_tags:
                print("=" * 70)
                print("C-terminus tags:")
                print("=" * 70)
                for tag in c_terminal_tags:
                    tag.info()
            if internal_tags:
                print("=" * 70)
                print("Internal tags:")
                print("=" * 70)
                for tag in internal_tags:
                    tag.info()

        return all_tags, termini

    def strip_tags(self, sequence: str, verbose: bool = True) -> (str, [ProteinTagHit]):
        """
        Stips tags from the N- and C-terminus of a sequence
        and reports the tags stripped away.

        Args:
            sequence: Input sequence to strip tags from.
            verbose:  If True, prints the identified tags.
        Returns:
            sequence, [ProteinTagHit]: A tuple of the stripped sequence and a
                                       list of removed tags that were stripped
                                       off.
        Raises:
            TypeError: If the input file is not a CSV file
        Example:
            >>> tf = TagFinder()
            >>> no_tag_seq, tag_list = tf.strip_tags.(sequence='GHHHHHHMYNAMEISCARL')
            >>> print(no_tag_seq)
            >>> 'MYNAMEISCARL'
            >>> print(tag_list[0].sequence)
            >>> 'GHHHHHH'
        """
        tags, termini = self.find_tags(sequence=sequence, verbose=verbose)
        stripped_seq = sequence[termini["N-terminus"][-1] : termini["C-terminus"][0]]
        removed_tags = [tag for tag in tags if tag.location != "Internal"]
        return stripped_seq, removed_tags

    def tag_strip_csv(self, input_file: str, output_file: str, col_name: str):
        """
        Strip terminal purification tags from an input .csv file
        containing protein sequences.

        Args:
            input_file (str):  Full path of the input CSV file.
            output_file (str): Full path of the output CSV file.
            col_name (str):    The column name containing the sequence.

        Returns:
            output_file (str): An output file where the sequences in
                               the column specified by 'col_name' have
                               their terminal tags stripped off.
                               A column called 'removed_tags' is added
                               to the output file with information about
                               the removed tags.
        Raises:
            TypeError: If the input file is not a CSV file
        Example:
            >>> tf = TagFinder()
            >>> tf.tag_strip_csv.(input_file='/path/to/my_input.csv',
            >>>                   output_file='/path/to/my_output.csv',
            >>>                   col_name='aa_seq')
        """
        if not input_file.endswith(".csv"):
            raise TypeError("Input File is not a .csv file!")

        dataframe = pd.read_csv(input_file, delimiter=",", low_memory=True)
        dataframe["removed_tags"] = dataframe.apply(lambda _: "", axis=1)
        tag_counter = 0
        seq_counter = 0
        tag_names = []
        tag_locations = []
        for i, row in tqdm(dataframe.iterrows(), total=dataframe.index.size, desc="Processing"):
            sequence = row[col_name]
            stripped_sequence, removed_tags = self.strip_tags(sequence, verbose=False)

            if len(removed_tags) != 0:
                tag_names += [tag.name for tag in removed_tags]
                tag_locations += [tag.location for tag in removed_tags]
                n_terminal_tags = [
                    f"{tag.name}({tag.sequence})"
                    for tag in removed_tags
                    if tag.location == "N-terminus"
                ]
                n_terminal_tags = (
                    "None" if len(n_terminal_tags) == 0 else " ".join(n_terminal_tags)
                )
                c_terminal_tags = [
                    f"{tag.name}({tag.sequence})"
                    for tag in removed_tags
                    if tag.location == "C-terminus"
                ]
                c_terminal_tags = (
                    "None" if len(c_terminal_tags) == 0 else " ".join(c_terminal_tags)
                )
                removal_report = f"N-term: {n_terminal_tags} C-term: {c_terminal_tags}"
                tag_counter += 1
            else:
                removal_report = f"N-term: None C-term: None"

            dataframe.at[i, col_name] = stripped_sequence
            dataframe.at[i, "removed_tags"] = removal_report
            seq_counter += 1

        # Generate basic report
        tag_percent = int(round(tag_counter / seq_counter * 100))
        tag_total = len(tag_locations)
        n_term_counter = tag_locations.count("N-terminus")
        n_term_percent = int(round(n_term_counter / tag_total * 100))
        c_term_counter = tag_locations.count("C-terminus")
        c_term_percent = int(round(c_term_counter / tag_total * 100))
        report = f"Found and removed tags in {tag_counter}/{seq_counter} sequences ({tag_percent}%)\n"
        report += "Tag locations:\n"
        report += f"N-terminus: {n_term_counter}/{tag_total} ({n_term_percent}%)\n"
        report += f"C-terminus: {c_term_counter}/{tag_total} ({c_term_percent}%)\n"
        report += "Tag types:\n"
        report += "\n".join(
            str(pd.DataFrame(tag_names)[0].value_counts()).split("\n")[1:-1]
        )
        print(report)
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    TF = TagFinder()
    df = pd.read_parquet(sys.argv[1])
    df.to_csv("test.csv")
    TF.tag_strip_csv(
        input_file="test.csv", output_file="no_tags.csv", col_name="aa_seq"
    )
