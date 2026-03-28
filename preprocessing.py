from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy



#=========================== TEXT CLEANING FUNCTION ====================================================================

def text_cleaning_func(text):
    lem = WordNetLemmatizer()
    text = text.lower()
    remove_email = re.sub(r"\b[a-zA-Z0-9.-_%+]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}", "", text)
    english_digit_words = " ".join(re.findall(r"(?u)\b\w\w+\b", remove_email))
    removing_digits = re.sub(r"\d+", "", english_digit_words)
    final_text = re.sub(r"\s+", " ", removing_digits)
    return final_text

#=========================== TF-IDF FUNCTION ==========================================================================
# TF-IDF: Text to Vector
tfidf = TfidfVectorizer(
    stop_words = "english",
    strip_accents = "ascii",
    token_pattern = r"\b\w\w+\b",
    analyzer = "word",
    lowercase = False,
    ngram_range = (1,1),
    max_df = 1.0,
    min_df = 1)

#=============================== SKILLS TYPES ==========================================================================
all_skills = [
    # ==================== Programming & Core Tech ====================
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust", "kotlin", "swift", "r", "matlab", "scala", "perl",
    "bash", "shell scripting", "dart", "objective-c",

    # ==================== Web Development ====================
    "html", "css", "bootstrap", "tailwind", "tailwind css",
    "react", "angular", "vue", "vue.js", "next.js", "nuxt.js", "svelte",
    "node.js", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "asp.net", "asp.net core", "laravel", "symfony", "ruby on rails",
    "rest api", "restful api", "graphql", "websockets", "soap", "jwt", "oauth",

    # ==================== Data Science & AI ====================
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data mining",
    "feature engineering", "model building", "model deployment",
    "predictive modeling", "statistics", "time series analysis",
    "reinforcement learning",

    # ==================== AI / GenAI ====================
    "generative ai", "llm", "large language models",
    "prompt engineering", "fine tuning", "rag",
    "retrieval augmented generation", "openai api", "openai",
    "gpt", "llama", "langchain", "hugging face",

    # ==================== AI/ML Libraries ====================
    "scikit-learn", "tensorflow", "keras", "pytorch",
    "xgboost", "lightgbm", "catboost",
    "opencv", "nltk", "spacy", "gensim",
    "transformers", "huggingface", "langchain", "pandas", "numpy",

    # ==================== Databases ====================
    "sql", "mysql", "postgresql", "postgres", "mongodb", "sqlite",
    "oracle", "redis", "cassandra", "dynamodb", "firebase",
    "vector databases", "faiss", "pinecone", "chroma", "elasticsearch",
    "neo4j", "mariadb", "couchdb",

    # ==================== Data Engineering ====================
    "apache spark", "hadoop", "kafka", "etl",
    "data pipeline", "airflow", "data warehousing",
    "bigquery", "snowflake", "spark streaming",
    "hive", "pig", "flink", "presto", "trino",

    # ==================== Cloud ====================
    "aws", "amazon web services", "azure", "google cloud", "gcp",
    "ec2", "s3", "lambda", "cloud functions", "cloud run", "cloud storage",
    "serverless architecture", "serverless",

    # ==================== DevOps ====================
    "docker", "kubernetes", "jenkins", "ci/cd",
    "github actions", "gitlab ci/cd", "terraform", "ansible",
    "prometheus", "grafana", "monitoring", "puppet", "chef",
    "blue green deployment", "canary deployment",
    "infrastructure as code", "auto scaling", "elk stack", "logstash", "kibana",

    # ==================== Backend Advanced ====================
    "api gateway", "rate limiting", "authentication",
    "authorization", "oauth", "jwt", "session management",
    "caching", "redis caching", "load balancing", "microservices",
    "event driven architecture", "message queues", "rabbitmq",

    # ==================== System Design ====================
    "scalability", "high availability", "fault tolerance",
    "distributed systems", "event driven architecture",
    "message queues", "pub/sub", "system design",

    # ==================== Networking ====================
    "tcp/ip", "dns", "http", "https",
    "network protocols", "firewalls", "vpn", "ssl/tls",

    # ==================== Mobile Development ====================
    "android", "ios", "react native", "flutter", "xamarin",
    "swiftui", "uikit", "jetpack compose", "kotlin multiplatform",

    # ==================== Testing ====================
    "unit testing", "integration testing", "pytest", "junit", "testng",
    "selenium", "test automation", "automated testing", "cypress",
    "performance testing", "jmeter", "manual testing", "quality assurance", "qa",

    # ==================== Tools ====================
    "git", "github", "gitlab", "bitbucket", "svn",
    "jira", "confluence", "postman", "swagger",
    "figma", "adobe xd", "sketch", "tableau", "power bi", "excel",
    "ms office", "powerpoint", "word",

    # ==================== Cybersecurity ====================
    "network security", "penetration testing", "ethical hacking",
    "cryptography", "owasp", "vulnerability assessment",
    "identity access management", "zero trust security",
    "siem", "threat modeling", "incident response", "compliance", "gdpr", "iso 27001",

    # ==================== Blockchain ====================
    "blockchain", "ethereum", "smart contracts",
    "solidity", "web3", "defi",

    # ==================== IoT / Robotics ====================
    "internet of things", "iot", "embedded systems",
    "raspberry pi", "arduino", "sensor integration",
    "robotics", "ros", "automation systems",

    # ==================== AR/VR ====================
    "augmented reality", "ar", "virtual reality", "vr", "mixed reality", "mr",

    # ==================== Game Development ====================
    "unity", "unreal engine", "game physics", "3d modeling",

    # ==================== Data & Analytics ====================
    "data visualization", "data cleaning", "data interpretation",
    "statistical analysis", "business intelligence", "bi",
    "forecasting", "a/b testing", "kpi tracking",
    "dashboarding", "business analytics",

    # ==================== Management ====================
    "project management", "program management", "product management",
    "people management", "team management",
    "stakeholder management", "resource management",
    "time management", "risk management",
    "change management", "operations management",
    "strategic planning", "decision making",

    # ==================== Agile ====================
    "agile", "scrum", "kanban", "sprint planning",
    "backlog grooming", "retrospectives",
    "lean", "six sigma", "process improvement", "sdlc", "itil",

    # ==================== Business ====================
    "business analysis", "market research",
    "competitive analysis", "business strategy",
    "growth strategy", "go-to-market strategy",
    "revenue growth", "cost optimization",
    "profitability analysis", "requirement gathering",

    # ==================== Finance ====================
    "financial analysis", "budgeting", "forecasting",
    "accounting", "taxation", "auditing",
    "financial modeling", "cost accounting",
    "balance sheet", "profit and loss",
    "investment analysis", "risk analysis",
    "credit analysis", "loan processing",
    "fraud detection", "fintech", "trading", "derivatives",

    # ==================== Marketing ====================
    "digital marketing", "seo", "sem", "search engine optimization", "search engine marketing",
    "social media marketing", "content marketing",
    "email marketing", "branding",
    "campaign management", "lead generation",
    "google analytics", "performance marketing",
    "market segmentation", "customer acquisition",
    "conversion optimization", "cart abandonment",

    # ==================== Sales ====================
    "sales strategy", "b2b sales", "b2c sales",
    "client relationship management", "crm",
    "negotiation", "lead conversion",
    "pipeline management", "account management", "salesforce",

    # ==================== HR ====================
    "recruitment", "talent acquisition",
    "employee engagement", "performance management",
    "payroll", "hr operations",
    "onboarding", "training and development",
    "conflict resolution", "policy management",

    # ==================== Customer Support ====================
    "customer service", "customer support",
    "client handling", "complaint resolution",
    "customer satisfaction", "call center operations",

    # ==================== Operations ====================
    "supply chain management", "logistics",
    "inventory management", "vendor management",
    "procurement", "quality control",
    "process optimization", "demand planning",
    "warehouse management", "last mile delivery",

    # ==================== Design ====================
    "ui design", "ux design", "graphic design",
    "adobe photoshop", "illustrator", "photoshop",
    "wireframing", "prototyping",
    "user research", "interaction design",
    "usability testing", "design systems",

    # ==================== Communication ====================
    "technical writing", "content writing",
    "copywriting", "documentation",
    "report writing", "presentation skills",
    "public speaking", "storytelling",

    # ==================== Soft Skills ====================
    "communication", "leadership", "teamwork", "collaboration",
    "problem solving", "critical thinking",
    "adaptability", "creativity", "collaboration",
    "emotional intelligence", "work ethic",
    "self motivated", "quick learner",
    "attention to detail", "multitasking",
    "ownership", "result oriented", "mentoring",

    # ==================== Research ====================
    "research", "academic writing",
    "data interpretation", "hypothesis testing",
    "literature review",

    # ==================== Healthcare ====================
    "patient care", "clinical research",
    "medical coding", "healthcare management",
    "ehr", "electronic health records",
    "hipaa compliance", "medical imaging",

    # ==================== Legal ====================
    "legal research", "compliance",
    "contract management", "regulatory affairs",
    "intellectual property", "contract drafting",
    "litigation support",

    # ==================== Retail / E-commerce ====================
    "merchandising", "store operations",
    "inventory control", "pos systems",
    "shopify", "woocommerce",

    # ==================== Hospitality ====================
    "hotel management", "front office operations",
    "guest relations",

    # ==================== Education ====================
    "curriculum development", "teaching",
    "mentoring", "instructional design",

    # ==================== Operating Systems ====================
    "linux", "unix", "windows", "macos", "ubuntu", "centos", "red hat",

    # ==================== Core CS ====================
    "oop", "object oriented programming", "data structures", "algorithms",
    "api development", "microservices", "system design",

    # ==================== Languages ====================
    "english", "hindi", "spanish",
    "french", "german", "mandarin"]

#=============================== Type OF EDUCATIONS LIST ================================================================
all_education = [
    # ==================== Computer Science & IT ====================
    "computer science", "computer science and engineering", "cse",
    "information technology", "it", "software engineering",
    "computer engineering", "computer applications", "bca", "mca",
    "data science", "artificial intelligence", "machine learning",
    "cybersecurity", "information security", "network security",
    "cloud computing", "data analytics", "big data",
    "computer information systems", "cis", "management information systems", "mis",
    "information systems", "computing", "informatics",

    # ==================== Degrees ====================
    "bachelor of technology", "b.tech", "btech",
    "bachelor of engineering", "b.e", "be",
    "bachelor of science", "b.sc", "bsc",
    "bachelor of arts", "ba",
    "bachelor of commerce", "b.com", "bcom",
    "bachelor of computer applications", "bca",
    "bachelor of business administration", "bba",
    "bachelor of pharmacy", "b.pharma", "bpharm",

    "master of technology", "m.tech", "mtech",
    "master of engineering", "m.e", "me",
    "master of science", "m.sc", "msc",
    "master of arts", "ma",
    "master of commerce", "m.com", "mcom",
    "master of computer applications", "mca",
    "master of business administration", "mba",
    "master of pharmacy", "m.pharma", "mpharm",

    "doctor of philosophy", "phd", "ph.d", "doctorate",

    # ==================== Engineering ====================
    "engineering", "bachelor of engineering", "be", "btech",
    "master of engineering", "me", "mtech",
    "electrical engineering", "electronics engineering", "ece",
    "electronics and communication engineering", "electrical and electronics engineering", "eee",
    "mechanical engineering", "civil engineering", "chemical engineering",
    "biotechnology", "biomedical engineering", "bioengineering",
    "aerospace engineering", "automotive engineering", "industrial engineering",
    "manufacturing engineering", "materials engineering", "metallurgical engineering",
    "petroleum engineering", "mining engineering", "environmental engineering",
    "agricultural engineering", "structural engineering", "construction engineering",
    "instrumentation engineering", "control engineering", "robotics engineering",
    "mechatronics", "automation engineering", "nanotechnology",

    # ==================== Business & Management ====================
    "business administration", "bachelor of business administration", "bba",
    "master of business administration", "mba",
    "executive mba", "emba", "pgdm",
    "management", "business management", "international business",
    "finance", "accounting", "commerce", "bcom", "mcom",
    "marketing", "human resources", "hr", "operations management",
    "supply chain management", "logistics", "business analytics",
    "entrepreneurship", "family business management", "retail management",
    "hospitality management", "hotel management", "tourism management",
    "event management", "sports management", "healthcare management",
    "public administration", "nonprofit management",

    # ==================== Science ====================
    "science", "bachelor of science", "bsc", "master of science", "msc",
    "physics", "chemistry", "mathematics", "applied mathematics",
    "statistics", "biology", "zoology", "botany", "biochemistry",
    "microbiology", "genetics", "molecular biology", "biotechnology",
    "environmental science", "geology", "geography", "earth science",
    "astronomy", "astrophysics", "material science", "food science",
    "forensic science", "agriculture", "horticulture", "veterinary science",

    # ==================== Arts & Humanities ====================
    "arts", "bachelor of arts", "ba", "master of arts", "ma",
    "english literature", "english", "literature", "linguistics",
    "history", "political science", "sociology", "psychology",
    "philosophy", "economics", "anthropology", "archaeology",
    "journalism", "mass communication", "media studies",
    "public relations", "advertising", "film studies",
    "fine arts", "visual arts", "performing arts", "music",
    "theatre", "dance", "graphic design", "fashion design",
    "interior design", "architecture", "urban planning",

    # ==================== Law ====================
    "law", "bachelor of laws", "llb", "master of laws", "llm",
    "juris doctor", "jd", "corporate law", "criminal law",
    "constitutional law", "intellectual property law", "tax law",
    "international law", "human rights law", "environmental law",

    # ==================== Healthcare & Medicine ====================
    "medicine", "mbbs", "doctor of medicine", "md",
    "bachelor of dental surgery", "bds", "master of dental surgery", "mds",
    "bachelor of ayurvedic medicine and surgery", "bams",
    "bachelor of homeopathic medicine and surgery", "bhms",
    "nursing", "bsc nursing", "msc nursing",
    "pharmacy", "bpharm", "mpharm", "pharmd",
    "physiotherapy", "occupational therapy", "speech therapy",
    "public health", "mph", "epidemiology",
    "clinical research", "health informatics", "health administration",
    "veterinary medicine", "bvsc", "mvsc",

    # ==================== Education ====================
    "education", "bachelor of education", "bed", "master of education", "med",
    "early childhood education", "elementary education", "secondary education",
    "special education", "educational psychology", "educational leadership",
    "curriculum and instruction", "educational technology",

    # ==================== Diplomas ====================
    "diploma", "polytechnic diploma", "polytechnic",
    "advanced diploma", "post graduate diploma", "pg diploma", "pgdm",
    "vocational training", "certificate",
    "computer applications", "dca", "pdc", "web development",
    "graphic design", "animation", "multimedia",
    "automobile engineering", "tool and die", "welding", "fabrication",
    "electrical technician", "electronics technician", "plumbing",
    "carpentry", "hvac", "air conditioning", "refrigeration",
    "beauty and wellness", "culinary arts", "bakery", "confectionery",

    # ==================== Social Sciences ====================
    "social work", "bachelor of social work", "bsw", "master of social work", "msw",
    "sociology", "anthropology", "criminology", "gender studies",
    "development studies", "international relations", "diplomacy",

    # ==================== Doctoral & Research ====================
    "doctor of philosophy", "phd", "ph.d", "doctorate",
    "post doctoral", "postdoctoral", "research fellowship",
    "dba", "doctor of business administration",
    "edd", "doctor of education", "dsci", "doctor of science",

    # ==================== Professional Certifications ====================
    "chartered accountant", "ca", "cpa", "certified public accountant",
    "company secretary", "cs", "cost accountant", "cma", "icwa",
    "chartered financial analyst", "cfa", "financial risk manager", "frm",
    "project management professional", "pmp", "prince2", "agile certification",
    "certified scrum master", "scrum master", "csm", "safe", "scaled agile",

    "aws certified solutions architect", "aws certification", "amazon web services certification",
    "azure certification", "microsoft certification", "microsoft azure certification",
    "google cloud certification", "google cloud platform certification", "gcp certification",
    "oracle certification", "cisco certification", "ccna", "ccnp", "ccie",

    "comptia", "a+", "network+", "security+",
    "certified ethical hacker", "ceh", "cissp", "cism", "cisa",
    "six sigma", "green belt", "black belt", "lean",
    "itil certification", "itil", "it service management",

    "data science certification", "machine learning certification",
    "artificial intelligence certification", "ai certification",
    "python certification", "java certification",

    "digital marketing certification", "google analytics certification",
    "teaching certification", "tefl", "tesol", "celt",
    "human resources certification", "shrm", "phr", "sphr",

    # ==================== Academic Levels ====================
    "high school", "secondary school", "higher secondary", "intermediate",
    "bachelor", "bachelors", "undergraduate", "ug",
    "master", "masters", "postgraduate", "pg",
    "doctoral", "doctorate", "postdoctoral",
    "associate degree", "community college",
    "post graduate diploma", "pg diploma", "graduate diploma",

    # ==================== Degree Abbreviations ====================
    "ba", "bsc", "bcom", "bca", "bba", "btech", "be", "barch", "bdes",
    "ma", "msc", "mcom", "mca", "mba", "mtech", "me", "march", "mdes",
    "llb", "llm", "phd", "ph.d", "dphil", "postdoc",
    "pgdm", "pgdba", "pgdca", "pgdscm",
    "b.ed", "m.ed", "b.p.ed", "m.p.ed",
    "b.pharm", "m.pharm", "b.d.s", "m.d.s", "b.v.sc", "m.v.sc",
    "bams", "bhms", "mbbs", "bds", "md", "ms"]

#============================== EXTRACTING PHONE_NUMBER AND EMAIL_ID, WORK_EXP.=========================================
def phone_number(text):
    return re.findall(r"(?:\+91[\s\-]?)?[6-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{4}", text)

def email_id(text):
    return re.findall(r"\b[a-zA-Z0-9.-_%+]+@[a-zA-Z0-9.-_]+\.[a-zA-Z]{2,}", text)

def extracting_work_exp(text):
    return " ".join(re.findall(r"\b\d{1,2}\s+years?\b", text) or re.findall(r"\b\d{1,2}\s+yrs?\b", text))

#============================== EXTRACTING SKILLS ======================================================================
def extract_skills(text):
    found = []
    for skill in all_skills:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)
    found = list(set(found))
    return ", ".join(found)

#============================== EXTRACTING EDUCATION ===================================================================
def extract_education(text):
    found = []
    for edu in all_education:
        if re.search(rf"\b{re.escape(edu)}\b", text):
            found.append(edu)
    found = list(set(found))
    return ", ".join(found)

#============================== EXTRACTING CANDIDATES NAME =============================================================
def extracting_name(text):
    nlp = spacy.load("en_core_web_sm")
    text = text[:100]
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""

#============================== EXTRACTING CANDIDATES NAME =============================================================
# skills score calculation functions
def extract_skills_str(text):
    found = []
    for skill in all_skills:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)
    found = list(set(found))
    return " ".join(found)

# Education score calculation functions
def extract_edu_str(text):
    found = []
    for skill in all_education:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.append(skill)
    found = list(set(found))
    return " ".join(found)











