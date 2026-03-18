from utils.met_utils import get_met_items, extract_met_image
from utils.reddit_utils import get_reddit_items, extract_reddit_image
from utils.rijks_utils import get_rijks_items, extract_rijks_image
from utils.wikiart_utils import get_wikiart_items, extract_wikiart_image



FUNCTION_REGISTRY = {
    "get_wikiart_items": get_wikiart_items,
    "extract_wikiart_image": extract_wikiart_image,
    "get_rijks_items": get_rijks_items,
    "extract_rijks_image": extract_rijks_image,
    "get_reddit_items": get_reddit_items,
    "extract_reddit_image": extract_reddit_image,
    "get_met_items": get_met_items,
    "extract_met_image": extract_met_image,
}