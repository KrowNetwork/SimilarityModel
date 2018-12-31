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
    sim = (sim - -1)/ (1 - -1)
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

job_1 = """Software Developer
Job ID 4258 Date posted 08/10/2018 Location NJ - Berkeley Heights

Job Description

BI&A is seeking a Software Developer to join their agile development team in Berkeley Heights, NJ.

JOB DUTIES INCLUDE:
• Interacts with customers, PMs and other development teams to gather, analyze and define requirements to determine the most effective software and web technologies to satisfy the client needs
• Develops, maintains, supports and enhances complex and diverse software systems (e.g., processing-intensive analytics, novel algorithm development, manipulation of extremely large data sets, real-time systems, and business management information systems) based upon documented requirements
• Provides specific input to the software components of system design to include hardware/software trade-offs, software reuse, and requirements analysis from system level to individual software components
• Utilizes software engineering and design methodologies appropriate to the development, integration, and enterprise level production environment
• Reviews and tests software components for adherence to the design requirements and documents test results
• Designs, creates, tests, and maintains software and web based applications and content solutions to satisfy customer requirements
• Follows a formal design process using formal specifications, data flow diagrams, and adheres to laws, standards, and established guidelines for development and delivery of software and web applications
• Designs and develops visually-pleasing, content rich, user-friendly interfaces with intuitive navigation
• Develops and maintains software and web development technical documentation to assist with software and web application maintenance and upgrades
• Provides software process management and configuration management throughout the software / web development lifecycle.
• Serves as the technical lead of multiple software development teams.
• Selects the software development processes in coordination with the customer and system engineering.
• Recommends new technologies and processes for complex software projects.
• Ensures quality control of all developed and modified software.
• Delegates programming and testing responsibilities to one or more teams and monitor their performance.
• Analyzes and troubleshoots extremely complex software problems and provides solutions using the latest technologies.
• Integrates new software and web products with existing software and web applications in order to improve the functionality or design of the system.

REQUIREMENTS:
• Experience in troubleshooting complex data analytic systems.
• Bachelor’s Degree with 8 years’ experience with enterprise level SDLC.
• Experience in Unix/Linux, Ruby/JRuby, Ruby on Rails, Python, Java, Java Script, C/C++, Shell Scripting, and ETL processes in a clustered environment.
• Technical understanding of the big data concepts, cloud technologies such as AWS, Hadoop, and strong trouble shooting skills.
• Applicant MUST have the ability to obtain Top Secret Clearance.

DESIRED SKILLS (although not required):

Familiarity with OAM concepts and Linux back end debugging skills are highly desired.

BI&A is an Equal Opportunity Employer. All qualified applicants will receive consideration for employment without regard to race,color, religion, sex, pregnancy, sexual orientation, gender identity, national origin, age, protected veteran status, or disability status.
"""

job_2 = """Job Description
Ref ID: 02720-0010771371

Classification: Software Engineer

Compensation: DOE

The successful candidate will have a passion for software development and the desire to constantly improve both their skills and the solutions they work on. This opening will provide the right individual the opportunity to work with some of the top people in the financial software industry on sophisticated design challenges. Our senior team includes individuals with years of experience in top-tier firms including Citigroup, Goldman Sachs, JP Morgan and Bear Stearns. Responsibilities ? Design, develop and support core technology services underlying Quantifi?s applications ? Identify and champion new technologies to extend Quantifi?s service-oriented architecture ? Work as a team member alongside other developers both in the U.S. and globally to implement software solutions
For immediate consideration, email all resumes to ryan.savage@rht.com

Job Requirements
? Bachelor's degree in a Computer Science-related discipline (Master?s degree a plus) ? Expert-level C#/.NET knowledge ? Experience with SQL and relational databases ? Excellent communications skills

Technology doesn't change the world. People do.

As a technology staffing firm, we can't think of a more fitting mantra. We're extreme believers in technology and the incredible things it can do. But we know that behind every smart piece of software, every powerful processor, and every brilliant line of code is an even more brilliant person.

Leader among IT staffing agencies

The intersection of technology and people — it's where we live. Backed by more than 65 years of experience, Robert Half Technology is a leader among IT staffing agencies. Whether you're looking to hire experienced technology talent or find the best technology jobs, we are your IT expert to call.

We understand not only the art of matching people, but also the science of technology. We use a proprietary matching tool that helps our staffing professionals connect just the right person to just the right job. And our network of industry connections and strategic partners remains unmatched.

Apply for this job now or contact our branch office at 888.674.2094 to learn more about this position.

All applicants applying for U.S. job openings must be authorized to work in the United States. All applicants applying for Canadian job openings must be authorized to work in Canada.

© 2018 Robert Half Technology. An Equal Opportunity Employer M/F/Disability/Veterans.

By clicking 'Apply Now' you are agreeing to Robert Half Terms of Use"""

job_3 = """Marketing / Sales / Business

--- Chat with a Recruiter Now!! https://flashrecruit.com/jobs/529006 ---

To join our marketing and sales team the perfect candidate would have great communication skills, possess a go-getter personality, and entrepreneurial spirit.

The position would start entry level, but the end goal is to cross train the person into management. In this role an individual would help increase sales and marketing services on behalf of our well known clients. How we do that is by meeting with their clients face to face. We offer the opportunity for growth and advancement only from within the company as our goal is to develop the best leaders within our organization.

Responsibilities:

Handle customer inquiries
Meet face to face with customers and conduct sales
Provide information about the products and services
Troubleshoot and resolve product issues and concerns
Document and update customer records based on interactions
Develop and maintain a knowledge base of the evolving products and services
Attend in office meetings
Complete spreadsheets for internal tracking​
Qualifications:

Previous experience in customer service, sales, or other related fields
Ability to build rapport with clients
Ability to prioritize and multitask
Positive and professional demeanor
Excellent written and verbal communication skills
Please note all positions are based in Bridgewater, NJ!

--- Chat with a Recruiter Now!! https://flashrecruit.com/jobs/529006 ---"""

docs = [job_1, job_2, job_3]
save_docs = ["Software Developer", "Software Engineer", "Marketer"]

docs = data_processor.clean(docs)[0]

## Split

docs = [d.split(" ") for d in docs]

## Create subwords

docs = [[data_processor.subword(w, w2v) for w in d] for d in docs]
sd = docs

## Pad

docs[0] = pad_sequences(docs[0], maxlen=25, value=len(w2v), padding="post")
docs[1] = pad_sequences(docs[1], maxlen=25, value=len(w2v), padding="post")
docs[2] = pad_sequences(docs[2], maxlen=25, value=len(w2v), padding="post")

# print (docs[0][:10])
# print ( )
# print (sd[0][:10])
# exit()
## Vecs

vecs = [0, 0, 0]

vecs[0] = sum(word2vec.predict(docs[0]))
vecs[1] = sum(word2vec.predict(docs[1]))
vecs[2] = sum(word2vec.predict(docs[2]))

print(vecs[0])
print(vecs[2])

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