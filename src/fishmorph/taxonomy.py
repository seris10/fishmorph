"""Name concepts only. No biological measurements or reference distributions are seeded."""
from .models import Taxon

SEED_TAXA = [
    Taxon(id="yellowstone_cutthroat", scientific_name="Oncorhynchus clarkii bouvieri",
          common_name="Yellowstone cutthroat trout", aliases=["Oncorhynchus virginalis bouvieri"],
          status="name_usage_conflict", profile_allowed=True,
          sources=["https://www.usgs.gov/publications/native-yellowstone-cutthroat-trout-oncorhynchus-virginalis-bouvieri-growth-and",
                   "https://explorer.natureserve.org/Taxon/ELEMENT_GLOBAL.2.102016/Oncorhynchus_virginalis_bouvieri"],
          note="Retains the brief's name as entered. Recent USGS publication uses O. virginalis bouvieri. Alias is a sourced name-usage note, not an automatic specimen reidentification. Checked 2026-10-13."),
    Taxon(id="mountain_whitefish", scientific_name="Prosopium williamsoni", common_name="Mountain whitefish",
          status="verified_name", profile_allowed=True,
          sources=["https://www.nps.gov/yell/learn/nature/fish-ecology.htm",
                   "https://onlinelibrary.wiley.com/doi/full/10.1002/tafs.10485"]),
    Taxon(id="arctic_grayling", scientific_name="Thymallus arcticus", common_name="Arctic grayling",
          status="verified_name", profile_allowed=True,
          sources=["https://www.nps.gov/yell/learn/nature/fish-ecology.htm",
                   "https://onlinelibrary.wiley.com/doi/full/10.1002/tafs.10485"]),
    Taxon(id="lake_trout", scientific_name="Salvelinus namaycush", common_name="Lake trout",
          status="verified_name", profile_allowed=True,
          sources=["https://scholarworks.montana.edu/items/ebe83d52-fb79-4a75-936d-59ceb7ce6737"]),
    Taxon(id="sculpin_unresolved", scientific_name="Cottus sp. (Yellowstone sculpin complex)",
          common_name="Unresolved Yellowstone sculpin", status="unresolved", profile_allowed=False,
          sources=["https://fieldguide.mt.gov/speciesDetail.aspx?elcode=AFC4E02440",
                   "https://nas.er.usgs.gov/queries/factsheet.aspx?SpeciesID=502"],
          note="Do not merge C. bondi, C. bairdii, or older Rocky Mountain usages by common name. Montana uses C. bondi for Rocky Mountain sculpin; this does not resolve every Yellowstone locality. Add a sourced identified concept before building profiles."),
]
