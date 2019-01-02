from keras.models import load_model, Model
import keras
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
import data_processor
import pickle 
import numpy as np
from keras import backend as K

np.set_printoptions(threshold=np.nan)

def calculate_similarity(v1, v2):
    sim = 1 - cosine(v1, v2)
    print (sim)
    # sim = (0.25 * sim) + (1.25 * (sim ** 2))
    sim = 5.5511150000000004e-17 + 0.375*sim + 0.9375*sim**2
    if sim > 1:
        sim = 0.99
    if sim < 0:
        sim = 0.01
    # sim = (sim - -1)/ (1 - -1)
    # print (type(sim))
    if sim == np.nan:
        sim = 0
    return sim

x = load_model("model.h5")
print ([layer.name for layer in x.layers])

# word2vec = load_model("w2v.h5")

word2vec = Model(inputs=x.input[0], output=x.get_layer("embedding").output)
# word2vec.save("test.h5")
# word2vec = load_model("test.h5")
# word2vec.compile(optimizer=keras.optimizers.Adam())
del x
# layer_output = get_3rd_layer_output([word1_p, word1_p])[0]

documents = pickle.load(open("clean.bin", "rb"))
# vocab = pickle.load(open("vocab.bin", "rb"))

# vocab = sorted(vocab)
# print (vocab[1000:])
# exit()

w2v = pickle.load(open("w2v.bin", "rb"))
# v2w = pickle.load(open("v2w.bin", "rb"))

job_1 = """Connected Devices Sales Consultant Serve on the sales floor, selling connected home devices, such as routers, smart home devices, audio equipment, and more. Responsible for building and maintaining customer relationships, and up selling services to benefit the customer.  Sales, Technology , Customer Support , Amazon Alexa , Audio Products , Google Home , Networking """

job_2 = """Seeing more than 200,000 patients per year, Mobile Health provides employment
related screening services to the New York City home health and health care
staffing industries and has established itself as the trusted vendor to almost
every home care agency in NYC –more than 700 clients.

A few key points:

· Mobile Health’s core offering is medical screenings for health care workers.
We perform over 120k drug screens, 100k tuberculosis tests and 60k physical
exams per year

· Our proprietary technology, which integrates with our clients’ HRIS systems,
supports the business

· Mobile Health has seven of its own clinics across NYC and hundreds of urgent
care center partners around the U.S. that provide services for its clients
outside the NYC catchment area

In 2018, Mobile Health introduced its online learning solution, Health
e-Class. With the long-term goal of sharing caregiver compliance information
between agencies, Mobile Health is using its monopolistic position and system
integrations to further entrench itself into its clients’ DNA. By the end of
2019, Mobile Health will have over 50,000 home health aides using its online
and integrated system to complete their annual in-service requirement.

Mobile Health is seeking a CTO to join its leadership team. This individual
will represent both Product and Technology, working closely with other leaders
and their respective teams, and be responsible for the Product Management,
IT/Helpdesk, Developer/Engineering, QA and DevOps/Infrastructure teams who, as
a group, are responsible for:

· Idealizing/Requirement gathering, scoping, ratifying, and facilitating the
ratification of products, enhancements and improvements.

· Architecting, development, quality, and deployment and support for all
platforms.

· Ensuring all platforms are scalable, stable, reliable and secure.

· Providing corporate IT functions, such as helpdesk, to ensure office
connectivity, computers, software, networks, printers, and phone systems
function properly.

· Delivering the development of precise and functional integrations between
integration partners and platforms.

The CTO will be the most senior IT position within Mobile Health. As such, the
organization will rely upon the successful candidate to help create our
technology roadmap, support existing products, and build a stable
infrastructure.

**Responsibilities:**

 **** · **Resource and Staff Management:** work with the leadership team to
properly define the roles and responsibilities of the technology team,
including organizational structure, job descriptions and team KPIs. Perform
vendor review to ensure all 3rd parties are known and working in conjunction
with our efforts.

· **Platform and Infrastructure Architecture:** working with vendors and the
entire IT team to document current platform architecture (short term task)
while, in parallel, collaborating with stakeholders to define and lay out
roadmap (longer term) and plan to reach desired architecture that supports
Mobile Health’s existing and future products.

· **Development Process and Communication:** Collaborate with business line
leaders in the product development process to ensure all work and projects are
linked to support the business goals and initiatives. Define and institute
best practices for both development and infrastructure efforts, including
improved communication up and out regarding various Initiatives and Projects.

· **Budget:** perform review of existing Budget and, working with Finance,
create annual departmental budget

**Skills/Competencies:**

To apply for this position, you must have the following skills and
competencies.

· **Languages and Libraries**

o C++

o C#

o Python

o Java

· **Applications &amp; Platforms**

o Microsoft IIS

o Microsoft SQL Server

o NoSQL Database

o Linux

o Windows

o Various caching and performance platforms (preferred, but not required)

· **Experience**

o Significant product, architect or VP experience architecting with proven
ability in building scalable technology infrastructure to support business
growth.

o Strong track record of 10-15+ years of progressive leadership with 2+ years
in a VP (or comparable) role, with experience in the health technology space.

o Experience working with highly sensitive data/PHI and installing systems to
protect said data

o Manage recruitment, development, and retention of team, keeping staff
focused, motivated and growing into future technical leaders.

o Work with various project and open source development tools, including, but
not limited to Slack, Jira, Confluence and GitHub.

o Experience in a start-up or highly entrepreneurial environment.

o Process oriented; able to build systemic metric efficiencies, which drive
speed and measure accountability, and manage multiple parallel development
work streams from concept to design and implementation and ongoing refinement

· **Personal**

o Entrepreneurial attitude.

o Business maturity beyond technical ability, with ability to prioritize
effectively in the face of business fluidity.

o Must be a leader, with ability to excel under pressure. Strong problem-
solving and motivational skills.

o Very creative, very flexible and thrives in an environment and industry that
perpetually evolves to meet business needs.

o Excellent written and oral communication skills, for technical and non-
technical audiences.

o Thinks out of the box, but rolls up sleeves."""

job_3 = """Autoholding 46 is a brand new high line used car facility in Mountain Lakes NJ! We have an opportunity for a highly motivated and experienced sales professionals looking to take a step into the luxury automobile environment. Sales Professionals can earn over $75,000 with commission and bonuses. The commission structure is based on the gross earned on the unit and bonuses are based on units sold per month.

Job Requirements:

Auto Sales Experience a plus

Must be a team player focused on providing a memorable customer experience
Must be self-motivated, have excellent attention to detail and have the drive and dedication to reach sales goals
Must be able to follow directions, have excellent organizational and time management skills, and the ability to multi-task
Sales Professionals should have a dynamic personality with a love of people and the need to please!
Must possess excellent communication skills, being able to articulate in person, via phone and internet with clients. Follow up skills are crucial to your success.
Must have a positive attitude, professional appearance, and be willing to go above and beyond to assist our clients.
Clean and Valid Driver’s License
Professionally present and demonstrate the automobile to clients, facilitating test drives, determining clients needs and wants, and presenting clients with purchase proposals
Answering phone and internet leads professionally and efficiently
Set and obtain monthly sales goals made by the sales professional and management
Obtain and maintain the Highest level of customer satisfaction and training levels to factory requirements
Ensure timely follow up and maintain strong relationships with previous and prospective clients.
Help maintain appearance and cleanliness of the dealership
Autoholding 46 is an Equal opportunity employer, drug free work place and great work environment.

Job Type: Commission

Experience:

sales: 1 year (Required)"""

docs = [job_1, job_2, job_3]
save_docs = ["Software Developer", "Software Engineer", "Sales"]

docs = data_processor.clean(docs)[0]

## Split

docs = [d.split(" ") for d in docs]

## Create subwords

docs = [[data_processor.subword(w, w2v) for w in d] for d in docs]
sd = docs

## Pad

docs[0] = pad_sequences(docs[0], maxlen=25, value=len(w2v), padding="post")
docs[1] = pad_sequences(docs[1], maxlen=25, value=len(w2v), padding="post")
# print (docs[1])
docs[2] = pad_sequences(docs[2], maxlen=25, value=len(w2v), padding="post")
# print (docs[2])
# print (docs[0][:10])
# print ( )
# print (sd[0][:10])
# exit()
## Vecs

vecs = [0, 0, 0]

x = word2vec.predict(docs[0])

for i in range(len(x)):
    a = x[i]
    a_d = np.linalg.norm(a)
    a = a / a_d
    x[i] = a

# x = np.array(vecs[0])
vecs[0] = sum(x)/len(x)
# print (vecs[0])
# print (np.array(vecs[0]).shape)
x = word2vec.predict(docs[1])
for i in range(len(x)):
    a = x[i]
    a_d = np.linalg.norm(a)
    a = a / a_d
    x[i] = a
vecs[1] = sum(x)/len(x)
x = word2vec.predict(docs[2])
for i in range(len(x)):
    a = x[i]
    a_d = np.linalg.norm(a)
    a = a / a_d
    x[i] = a
vecs[2] = sum(x)/len(x)

print (1/np.linalg.norm(vecs[0]-vecs[1]))
print (1/np.linalg.norm(vecs[1]-vecs[2]))
print (1/np.linalg.norm(vecs[2]-vecs[0]))

# print(vecs[0])
# print(vecs[2])

## Sim

sim = [0, 0, 0]

sim[0] = calculate_similarity(vecs[0], vecs[1])
sim[1] = calculate_similarity(vecs[1], vecs[2])
sim[2] = calculate_similarity(vecs[2], vecs[0])

## Display

print ("%s | %s: %s" % (save_docs[0], save_docs[1], sim[0]))
print ("%s | %s: %s" % (save_docs[1], save_docs[2], sim[1]))
print ("%s | %s: %s" % (save_docs[2], save_docs[0], sim[2]))

# w2v_w1 = word2vec.predict([word1_p])
# w2v_w2 = word2vec.predict([word2_p])
# w2v_w3 = word2vec.predict([word3_p])

# print (w2v_w1)
# print (w2v_w2)
# print (w2v_w3)

# print (calculate_similarity(w2v_w1, w2v_w2))
# print (calculate_similarity(w2v_w1, w2v_w3))
# print (calculate_similarity(w2v_w2, w2v_w3))

# # print (w2v_w1)
# # print (layer_output)