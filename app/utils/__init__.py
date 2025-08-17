from .date_utils import normalize_date_to_mmddyyyy, parse_date_to_unix

def parse_size(text: str):
	"""Parse sizes like '8 oz', '1.5 lb', '500 g', '12 fl oz'. Returns (value, unit) or (None, None)."""
	try:
		import re
		s = (text or "").lower()
		m = re.search(r"(\d+[\./]?\d*)\s*(fl\s*oz|oz|lb|g|kg|ml|l)", s)
		if not m:
			return (None, None)
		val = m.group(1).replace('/', '.')
		try:
			v = float(val)
		except Exception:
			v = None
		unit = m.group(2).replace(' ', '')
		return (v, unit)
	except Exception:
		return (None, None)


def normalize_abstract_name(name: str) -> str:
	"""Heuristic to canonicalize a product into an abstract concept (e.g., 'peanut butter')."""
	s = (name or "").strip()
	if not s:
		return ""
	s = s.lower()
	# remove non-alpha-only tokens like size markers
	tokens = [t for t in s.split() if any(c.isalpha() for c in t)]
	# remove common unit words
	units = {"oz","lb","g","kg","ml","l","fl","floz"}
	tokens = [t for t in tokens if t not in units]
	# naive stopwords
	stop = {"organic","creamy","crunchy","large","small","pack","of","the","and"}
	core = [t for t in tokens if t not in stop]
	if len(core) >= 2:
		return " ".join(core[-2:])
	if core:
		return core[0]
	return " ".join(tokens[-2:]) if len(tokens) >= 2 else (tokens[0] if tokens else s)
