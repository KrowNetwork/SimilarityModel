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

job_1 = """cognizant seeks an experienced Full Stack Developer to be part our customers in their journey to implement enterprise wide transformational projects. You will own the responsibility for designing and developing technical solutions for business requirements. Along this path, you will advise on the best practices and offer seasoned guidance on the technical solutions to meet the objectives of the project. As a key member of the project team, you will be interacting with the Project Manager, Techno Functional Analyst and Technical architect on a daily basis and will have a major influence on Cognizant’s relationship with the client.


Roles & Responsibilities:

Experience using backend technologies such as Java, Spring, Spring Boot, XML, Oracle Database.

Strong background in data structures and algorithms and computer science fundamentals.

Strong hands-on experience with REST Services

Build the solution construct in Cloud environment and assist with any migration elements.

Excellent skills in Typescripts, AngularJS/ReactJS, NodeJS, Javascript design architecture.

Strong Web development experience.

Strong Object Oriented experience and knowledge.

Ability to work collaboratively in teams

Hands-on experience designing and building large scale web/desktop and mobile platforms which are systems of record or integrate multiple related systems of record.

Qualification:

Any Bachelor’s degree with minimum 5+ years of experience as a Full stack developer is a plus


We are looking for applicants who have a flair for technology and are willing to take up challenging assignments.


**Applicants are required to be eligible to lawfully work in the U.S. immediately; employer will not sponsor applicants for U.S. work authorization (such as an H-1B visa) for this opportunity.


Technical Skills
SNo
Primary Skill
Proficiency Level *
Rqrd./Dsrd.

1
React
PL3
Desired

2
Node js
PL2
Desired

3
Backbone
PL2
Desired

4
Angular JS
PL2
Required

5
HTML 5
PL3
Required

6
JQuery
PL3
Desired

7
Advanced JavaScript
PL2
Required

8
JavaScript
PL2
Required

9
CSS
PL3
Required

Proficiency Legends
Proficiency Level
Generic Reference

PL1
The associate has basic awareness and comprehension of the skill and is in the process of acquiring this skill through various channels.

PL2
The associate possesses working knowledge of the skill, and can actively and independently apply this skill in engagements and projects.

PL3
The associate has comprehensive, in-depth and specialized knowledge of the skill. She / he has extensively demonstrated successful application of the skill in engagements or projects.

PL4
The associate can function as a subject matter expert for this skill. The associate is capable of analyzing, evaluating and synthesizing solutions using the skill.


Employee Status : Full Time Employee

Shift : Day Job

Travel : No

Job Posting : Dec 14 2018

Cognizant US Corporation is an Equal Opportunity Employer Minority/Female/Disability/Veteran. If you require accessibility assistance applying for open positions in the US, please send an email with your request to CareersNA2@cognizant.com
"""

job_2 = """We are looking for an experienced and talented software engineer to join our growing team in the Princeton area and build technology solutions in the healthcare industry. We are looking for software engineers to design custom software on a variety of platforms.

Requirements


5 years or more of actual work experience designing and architecting commercial software
Experience with the .NET framework and web application platforms (MVC, ASP.net)
Basic experience with databases is required, but more advanced experience with index plans, query plans, and writing SQL is a plus
Must work on premise in the Princeton, NJ area, please do not apply to work remotely
W2 Employment only, please no sub-contractors or recruiters
Comfortable in a team environment, i.e. using source control (SVN / Git) and working in an Agile environment
Healthcare industry experience and experience with HIPAA protected data is a plus
Benefits

We are not a recruiter or a placement agency - our team members are salaried, full-time W2 employees. You will not be placed at another company and will not work on-site with a customer. Work on-premise in our Princeton office.


Competitive salary
Fully vested 401k plan with employer matching contribution (up to 4% of salary)
Health and dental insurance, including company-funded FSA (flexible-spending account)
Paid time off for vacation and sick days"""

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