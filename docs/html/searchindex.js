Search.setIndex({"docnames": ["00_ai24/_build/jupyter_execute/lossFunc/rmse", "00_ai24/cv/convolution/cnnFrqExample", "00_ai24/cv/convolution/fullyConnectedDrawback", "00_ai24/cv/convolution/layers", "00_ai24/cv/convolution/pooling", "00_ai24/cv/convolution/readme", "00_ai24/gen", "00_ai24/lossFunc/readme", "00_ai24/lossFunc/rmse", "dataSets", "read", "readme", "youtubeExplained"], "filenames": ["00_ai24/_build/jupyter_execute/lossFunc/rmse.ipynb", "00_ai24/cv/convolution/cnnFrqExample.md", "00_ai24/cv/convolution/fullyConnectedDrawback.md", "00_ai24/cv/convolution/layers.md", "00_ai24/cv/convolution/pooling.md", "00_ai24/cv/convolution/readme.md", "00_ai24/gen.md", "00_ai24/lossFunc/readme.md", "00_ai24/lossFunc/rmse.ipynb", "dataSets.md", "read.md", "readme.md", "youtubeExplained.md"], "titles": ["Root Mean Square Error (RMSE)", "CNN for Frequency Estimation Problem", "Fully Connected Drawbacks", "Layers in Convolutional Neural Networks", "Pooling in Convolutional Neural Networks", "Convolution in Deep Learning", "&lt;no title&gt;", "loss", "Root Mean Square Error (RMSE)", "mnist", "read", "solai", "explained"], "terms": {"wide": [0, 8], "us": [0, 1, 2, 3, 4, 5, 8], "metric": [0, 1, 8], "measur": [0, 1, 8], "accuraci": [0, 8], "predict": [0, 1, 2, 8], "model": [0, 2, 8], "It": [0, 5, 8], "repres": [0, 8], "averag": [0, 1, 8], "differ": [0, 1, 3, 5, 8], "between": [0, 1, 2, 5, 8], "valu": [0, 4, 5, 8], "actual": [0, 1, 8], "here": [0, 8], "s": [0, 1, 2, 8], "breakdown": [0, 8], "what": [0, 8], "thi": [0, 1, 2, 4, 8], "calcul": [0, 3, 8], "For": [0, 1, 2, 8], "each": [0, 1, 2, 3, 4, 5, 8], "made": [0, 8], "known": [0, 2, 5, 8], "ensur": [0, 8], "all": [0, 2, 8], "ar": [0, 1, 2, 3, 5, 8], "posit": [0, 8], "emphas": [0, 8], "larger": [0, 8], "more": [0, 1, 2, 4, 5, 8], "than": [0, 2, 5, 8], "smaller": [0, 8], "ones": [0, 4, 5, 8], "comput": [0, 1, 4, 8], "give": [0, 8], "an": [0, 2, 3, 5, 8], "overal": [0, 8], "mse": [0, 8], "final": [0, 1, 8], "take": [0, 2, 8], "return": [0, 1, 8], "origin": [0, 8], "unit": [0, 8], "The": [0, 1, 3, 4, 5, 8], "text": [0, 1, 2, 8], "sqrt": [0, 1, 8], "frac": [0, 1, 8], "1": [0, 1, 8], "n": [0, 1, 8], "sum_": [0, 1, 8], "i": [0, 1, 8], "y_i": [0, 1, 8], "hat": [0, 1, 8], "y": [0, 1, 8], "_i": [0, 1, 8], "2": [0, 1, 8], "where": [0, 1, 2, 3, 4, 5, 8], "number": [0, 1, 2, 4, 8], "observ": [0, 8], "consist": [0, 1, 2, 8], "ha": [0, 1, 8], "same": [0, 1, 5, 8], "make": [0, 1, 2, 4, 8], "easier": [0, 8], "interpret": [0, 8], "compar": [0, 2, 4, 8], "sensit": [0, 8], "outlier": [0, 8], "due": [0, 8], "particularli": [0, 1, 5, 8], "larg": [0, 2, 8], "have": [0, 2, 8], "disproportion": [0, 8], "impact": [0, 8], "evalu": [0, 1, 8], "A": [0, 1, 5, 8], "lower": [0, 4, 8], "indic": [0, 1, 5, 8], "better": [0, 2, 8], "fit": [0, 1, 8], "data": [0, 1, 3, 4, 5, 8], "closer": [0, 8], "crucial": [0, 2, 4, 8], "assess": [0, 8], "perform": [0, 1, 2, 5, 8], "regress": [0, 1, 8], "provid": [0, 1, 8], "clear": [0, 1, 8], "how": [0, 1, 8], "well": [0, 2, 8], "match": [0, 8], "explor": 1, "basic": 1, "convolut": 1, "neural": [1, 5], "network": [1, 5], "solv": 1, "type": [1, 2], "involv": [1, 3, 5], "signal": [1, 5], "from": [1, 2, 3, 4, 5], "input": [1, 2, 3, 4, 5], "which": [1, 2, 4, 5], "common": [1, 5], "process": [1, 2, 3, 5], "time": [1, 2, 5], "seri": [1, 2, 5], "analysi": [1, 2, 5], "given": 1, "goal": 1, "applic": [1, 2], "like": [1, 2], "audio": 1, "identifi": [1, 5], "can": [1, 2, 5], "help": [1, 4, 5], "recogn": 1, "music": 1, "note": 1, "speech": [1, 2], "other": 1, "task": [1, 2], "typic": 1, "follow": [1, 3], "layer": [1, 5], "accept": 1, "raw": [1, 3], "simplic": 1, "assum": 1, "1d": 1, "arrai": [1, 5], "These": [1, 2, 3], "appli": [1, 3, 5], "filter": [1, 2], "extract": [1, 3, 4], "relev": 1, "featur": [1, 2, 3, 4, 5], "slide": [1, 2], "over": [1, 2, 5], "detect": [1, 2, 5], "pattern": [1, 2, 3, 5], "configur": 1, "includ": [1, 2], "determin": 1, "mani": 1, "learn": [1, 2, 10, 11], "size": [1, 2, 5], "defin": [1, 5], "stride": 1, "specifi": 1, "much": 1, "move": [1, 3], "step": [1, 2], "pad": 1, "decid": 1, "whether": 1, "maintain": [1, 2, 5], "output": [1, 2, 4, 5], "pool": [1, 3], "reduc": [1, 2, 4, 5], "dimension": [1, 2, 4], "map": [1, 3, 4, 5], "max": 1, "complex": 1, "domin": 1, "fulli": 1, "connect": 1, "after": 1, "sever": [1, 3], "flatten": 1, "pass": 1, "through": [1, 3, 5], "one": [1, 2], "act": 1, "classifi": 1, "regressor": 1, "depend": [1, 2], "neuron": [1, 2], "linear": [1, 3, 4], "activ": [1, 3], "root": 1, "mean": [1, 2, 4], "squar": 1, "error": 1, "rmse": 1, "magnitud": 1, "sampl": 1, "below": 1, "simpl": [1, 4], "python": 1, "kera": 1, "build": [1, 2], "train": [1, 2, 3, 5], "import": [1, 2, 4, 5], "numpi": [1, 5], "np": [1, 5], "tensorflow": [1, 10], "tf": 1, "sequenti": [1, 2], "conv1d": 1, "maxpooling1d": 1, "dens": [1, 2], "meansquarederror": 1, "gener": [1, 2], "synthet": 1, "def": 1, "generate_synthetic_data": 1, "num_sampl": 1, "length": 1, "x": [1, 3], "random": 1, "randn": 1, "uniform": 1, "low": 1, "0": [1, 5], "high": [1, 2], "10": [1, 5], "prepar": 1, "1000": 1, "100": 1, "x_train": 1, "y_train": 1, "x_test": 1, "y_test": 1, "32": 1, "kernel_s": 1, "3": 1, "relu": 1, "input_shap": 1, "pool_siz": 1, "64": 1, "50": 1, "compil": 1, "optim": 1, "adam": 1, "rootmeansquarederror": 1, "epoch": 1, "batch_siz": 1, "validation_split": 1, "print": [1, 5], "f": 1, "test": 1, "new": [1, 2, 3], "5": [1, 5], "simul": 1, "nois": [1, 5], "uniformli": 1, "distribut": [1, 2], "built": 1, "two": 1, "produc": [1, 3, 5], "set": [1, 3], "demonstr": 1, "its": 1, "capabl": 1, "foundat": 1, "understand": [1, 2], "paramet": [1, 2, 3, 4], "further": 1, "tune": 1, "base": 1, "specif": [1, 2, 5], "requir": [1, 2], "characterist": 1, "also": [2, 5], "fundament": [2, 5], "compon": 2, "everi": 2, "next": 2, "despit": 2, "util": 2, "thei": 2, "come": 2, "signific": [2, 4], "increas": 2, "lead": [2, 5], "overfit": [2, 4], "poorli": 2, "unseen": 2, "computation": 2, "expens": 2, "intens": [2, 5], "fix": 2, "cannot": 2, "handl": 2, "variabl": 2, "limit": 2, "problemat": 2, "imag": [2, 3, 5], "vari": 2, "lack": 2, "emphasi": 2, "do": 2, "account": 2, "spatial": [2, 4], "tempor": 2, "exampl": [2, 4], "nearbi": 2, "pixel": 2, "often": [2, 5], "relat": 2, "treat": 2, "independ": 2, "ignor": 2, "abil": 2, "effect": [2, 4, 5], "improv": [2, 4], "leverag": 2, "significantli": 2, "outperform": 2, "naiv": 2, "implement": 2, "stcuct": 2, "struct": 2, "video": 2, "spatila": 2, "design": 2, "By": [2, 4], "captur": [2, 3], "local": [2, 5], "edg": [2, 3, 5], "textur": [2, 3, 5], "hierarch": 2, "represent": [2, 4], "approach": 2, "enhanc": 2, "purpos": 2, "hidden": 2, "state": 2, "inform": [2, 4], "previou": 2, "natur": 2, "languag": 2, "recognit": 2, "advantag": 2, "address": 2, "gate": 2, "forget": 2, "control": 2, "flow": 2, "similar": 2, "longer": 2, "sequenc": 2, "mitig": [2, 4], "vanish": 2, "gradient": [2, 3], "problem": 2, "allow": 2, "node": 2, "relationship": 2, "them": 2, "social": 2, "recommend": 2, "system": 2, "bioinformat": 2, "point": 2, "focu": 2, "part": 2, "self": 2, "parallel": 2, "e": [2, 3], "g": [2, 3], "translat": [2, 4], "effici": [2, 4], "across": 2, "entir": 2, "hierarchi": 2, "tradit": [2, 5], "group": 2, "variou": [2, 3], "properti": 2, "object": 2, "especi": [2, 5], "pose": 2, "preserv": 2, "robust": 2, "affin": 2, "encod": 2, "compress": 2, "decod": 2, "reconstruct": 2, "reduct": [2, 4], "denois": 2, "anomali": 2, "most": [2, 4, 5], "enabl": 2, "probabilist": 2, "underli": 2, "realist": 2, "techniqu": 2, "within": [2, 4, 5], "tailor": 2, "In": [3, 5], "cnn": [3, 4, 5], "simultan": 3, "dure": 3, "transform": [3, 5], "oper": [3, 4, 5], "option": 3, "addit": 3, "non": [3, 4], "both": 3, "kernel": [3, 5], "bias": 3, "function": 3, "convolv": [3, 5], "term": 3, "To": 3, "add": [3, 5], "replac": 3, "h_i": 3, "b": 3, "backpropag": 3, "loss": 3, "respect": 3, "updat": 3, "minim": 3, "dimens": 4, "decreas": 4, "thu": 4, "risk": 4, "element": [4, 5], "window": 4, "becaus": [4, 5], "retain": 4, "diagram": 4, "show": 4, "2x2": 4, "correspond": 4, "region": [4, 5], "maximum": 4, "select": 4, "promin": 4, "discard": 4, "less": 4, "cost": 4, "while": 4, "invari": 4, "small": 4, "distort": 4, "highlight": 5, "definit": 5, "toeplitz": 5, "special": 5, "kind": 5, "descend": 5, "diagon": 5, "left": 5, "right": 5, "constant": 5, "fill": 5, "multipl": 5, "zero": 5, "result": 5, "higher": 5, "abstract": 5, "flip": 5, "befor": 5, "unlik": 5, "weight": 5, "directli": 5, "correl": 5, "rather": 5, "true": 5, "align": 5, "should": 5, "center": 5, "If": 5, "even": 5, "method": 5, "usual": 5, "replic": 5, "circular": 5, "wrap": 5, "around": 5, "were": 5, "symmetr": 5, "mirror": 5, "reflect": 5, "boundari": 5, "extend": 5, "descript": 5, "smooth": 5, "adjac": 5, "trend": 5, "rate": 5, "chang": 5, "rapid": 5, "presenc": 5, "noisi": 5, "input_sign": 5, "4": 5, "6": 5, "7": 5, "8": 5, "9": 5, "output_sign": 5, "mode": 5, "three": 5, "transit": 5, "r": 6, "http": [9, 10, 12], "yann": 9, "lecun": 9, "com": [9, 12], "exdb": 9, "www": [10, 12], "org": 10, "resourc": 10, "ml": 10, "deeplearningbook": 10, "explain": 10, "ai": 10, "repo": 11, "3blue1brown": 12, "youtub": 12, "playlist": 12, "list": 12, "plzhqobowtqdnu6r1_67000dx_zcjb": 12, "3pi": 12}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"root": [0, 8], "mean": [0, 8], "squar": [0, 8], "error": [0, 8], "rmse": [0, 8], "understand": [0, 8], "mathemat": [0, 8], "represent": [0, 8], "kei": [0, 5, 8], "point": [0, 8], "conclus": [0, 8], "cnn": 1, "frequenc": 1, "estim": 1, "problem": 1, "descript": 1, "model": 1, "architectur": 1, "loss": [1, 7], "function": 1, "exampl": [1, 5], "code": 1, "explan": [1, 5], "fulli": 2, "connect": 2, "drawback": 2, "intern": 2, "structur": 2, "modul": 2, "exploit": 2, "data": 2, "convolut": [2, 3, 4, 5], "layer": [2, 3], "recurr": 2, "neural": [2, 3, 4], "network": [2, 3, 4], "rnn": 2, "overview": [2, 3, 4, 5], "long": 2, "short": 2, "term": 2, "memori": 2, "lstm": 2, "graph": 2, "gnn": 2, "attent": 2, "mechan": 2, "transform": 2, "capsul": 2, "autoencod": 2, "variat": 2, "vae": 2, "learn": [3, 5], "multipl": 3, "filter": [3, 5], "forward": 3, "pass": 3, "first": 3, "second": 3, "ad": 3, "bia": 3, "backward": 3, "pool": 4, "type": [4, 5], "averag": [4, 5], "max": 4, "benefit": 4, "deep": 5, "concept": 5, "toplitz": 5, "matrix": 5, "spars": 5, "comput": 5, "practic": 5, "note": 5, "pad": 5, "applic": 5, "move": 5, "deriv": 5, "match": 5, "1": 5, "2": 5, "3": 5, "mnist": 9, "read": 10, "solai": 11, "about": 11, "topic": 11, "explain": 12}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})