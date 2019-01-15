# from keras.models import load_model, Model
# import keras
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
import data_processor
import pickle 
import numpy as np
import nltk
# from keras import backend as K
import tensorflow as tf 
from commonregex import CommonRegex

K = tf.keras.backend
load_model = tf.keras.models.load_model
# Model = 


np.set_printoptions(threshold=np.nan)

def calculate_similarity(v1, v2):
    sim = 1 - cosine(v1, v2)
    # print (sim)
    # sim = (0.25 * sim) + (1.25 * (sim ** 2))
    # sim = 1.0288 + (0.0002860917 - 1.0288)/(1 + (sim/0.7662593)**14.08159)
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

word2vec = tf.keras.models.Model(inputs=x.input[0], outputs=x.get_layer("embedding").output)
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

job_2 = """What does a Best Buy Sales Consultant – Computing and DI do?

At Best Buy our mission is to leverage the unique talents and passions of our employees to inspire, delight, and enrich the lives our customers through technology and all its possibilities. If you have a passion and curiosity for what is possible and enjoy people, we invite you to join us on this mission.

Best Buy Sales Consultants excel at selling products and services, working closely with other members of the sales team. They accumulate the appropriate knowledge and expertise through continuous learning and self development. Then, armed with the right tools and knowledge, they create ease and add value to the Best Buy shopping experience, ensuring customers’ end-to-end needs are met. As a trusted advisor and partner, Best Buy Sales Consultants deliver unique customer value by developing strong relationships with customers, bringing them a little closer to family and friends by helping them close the gaps with technology.

You’re a techno learner who enjoys working with people-employees and customers! Sales Consultants must not only know latest products-mobile devices, laptops, tablets-but stay ahead of the curve by frequently exploring and learning about new products and solutions.

80% of your time you will:

Engage customers using selling skills to build complex, connected solutions in a fast-paced, dynamic environment where customers feel supported and leave delighted.

Inspire customers by showing them what’s possible with technology.


20% of your time you will:

Use innovative training tools to stay current, confident and complete, driving profitable growth and achieving individual and department goals.

Accumulate and apply the appropriate knowledge and expertise through continuous learning and self-development, enabling you to provide an excellent customer shopping experience.

Maintain your department’s merchandising and readiness to serve customers. Back up the sales team for phone and store pickup.

Perform other duties as assigned.


What are the Professional Requirements of a Best Buy Sales Consultant?

Basic Qualifications

Ability to work successfully as part of a team

Ability to work a flexible schedule inclusive of holidays, nights and weekends

Preferred Qualifications

3 months experience working in customer service, sales or related fields


What are my rewards and benefits?
Discover your career here! At Best Buy we offer much more than a paycheck. Surrounded by the latest and greatest technology, a team of amazing coworkers and a work environment where anything is possible, you’ll find it easy to be your best when you work with us. We provide an exciting work environment with a community of techno learners where you can be yourself while investing in your career. Empowered with knowledge you will discover endless opportunities to grow. From deep employee discounts to tuition reimbursement, to health, wealth and wellness benefits, to learning and development programs, we believe the success of our company depends on the passion of employees for learning, technology and people."""

job_1 = """Tucker Siegel
Co-founder and Chief
Technology Officer
Tucker Siegel
266 E Dudley Avenue
Westfield, NJ 07090
908.419.9415
tuckers@krow.network
siegel.tucker@gmail.com
Skills
Full stack developer with experience in languages ranging from
Javascript to Python, as well as deep learning and artificial intelligence
algorithms. Extensive experience in customer service and customer
relations in a retail and services setting.
Experience Krow Network / Co-founder and Chief Technology Officer
August 2018 - PRESENT, NEWARK, NJ
Leading multi-platform development to build a useable product for
job seekers in the United States. Oversee all development of user
facing and back-end services, and fully operate all artificial
intelligence programs.
Best Buy / Connected Devices Sales Consultant
OCTOBER 2018 - PRESENT, UNION, NJ
Develop relationships with customers to understand their situation and
recommend the right solution. Implement sales tactics as well as sell
recurring services and sign ups for the store credit card.
Westfield, NJ /Intern at Technology Advisory Committee
OCTOBER 2018 - PRESENT, WESTFIELD, NJ
Meet with team members once a month to develop new technological
solutions to current issues facing the town. Advise the mayor on
budgeting concerns, and implementation of new ideas.
Gold Medal Fitness / Member Services Representative
MAY 2017 - PRESENT, GARWOOD, NJ
Serve at the front desk for Gold Medal Fitness at the Garwood and
Cranford, NJ locations. Responsible for handling member complaints,
snack and beverage sales, and enrolling new members.
ㅡ
Education
Westfield High School
CLASS OF 2019
Graduating with extensive classes in the STEM realm, and AP classes
ranging from AP Statistics to AP Computer Science. Involved as Vice
President of the STEM Club, as well as in the band program and multiple
varsity sports.
ㅡ
Volunteer
Westfield Presbyterian Church / Student Volunteer at Agape
Community Kitchen
SEPTEMBER 2012 - PRESENT, ELIZABETH, NJ
Involved with providing a comfortable experience for the guests who
come to the kitchen. Usually leading drinks preparation/serving or food
preparation, and guiding newer members on how to do each job at the
kitchen.
Presbyterian Disaster Relief/ Student Volunteer
JUNE 2017 - PRESENT
Serve as student volunteer on two service trips, one to North Carolina
after Hurricane Matthew, and one to Houston after Hurricane Harvey.
Help rebuild damaged property, and create protective barriers incase
another storm returns to the area."""

parse = CommonRegex(job_2)
print (parse)
print (parse.phones)
for i in parse.phones:
    job_2 = job_2.replace(i, "")

for i in parse.emails:
    job_2 = job_2.replace(i, "")

for i in parse.street_addresses:
    job_2 = job_2.replace(i, "")


job_3 = """We are looking for an experienced and responsible landscaper to join our team. The ideal candidate will be comfortable operating lawn maintenance equipment such as lawn mowers, trimmers, and blowers.

The landscaper will ensure the growth and vibrancy of our plants, flowers, lawn, and decorative shrubs. The landscaper will water, fertilize, and prune to remove damaged or dying plant life. The candidate can expect to work outdoors in a mix of weather conditions, and be able to perform maintenance duties to ensure employee safety during inclement weather by removing debris, snow, and ice from communal walkways and spaces.

Landscaper Duties and Responsibilities
Operate push or riding lawnmowers; may operate heavier tractor equipment if needed
Water all plants and lawn, and ensure all plants are evenly covered
Spread fertilizer, plant food, mulch, and other materials around plants
Remove weeds and dead plants; prune overgrown limbs and leaves
Operate string trimmer and edger to remove overgrowth and keep outdoor area tidy
Use leaf blower to clear walkways and pedestrian areas after lawn maintenance
Treat lawn and landscaping with pesticides to remove harmful insects
Maintain the existing landscaping design and ensure plant survival
Plant new flowers, bushes, plants, and decorative shrubs
Rake fallen leaves and remove debris
Keep pedestrian areas removed of snow and ice
Operate heavier snow blowers or other equipment as needed
Remove tree limbs, overgrowth, and other hazards
Ensure outdoor furniture, décor, and lighting is maintained and good working order
Properly store and handle all equipment, tools, sprinklers, etc.
Oversee maintenance repairs to equipment, landscape structures, and hardscape walkways
Landscaper Requirements and Qualifications
High school degree or equivalent educational experience
Previous experience in landscaping or groundskeeping a plus
Must pass background check
Must be 18 years of age
Able to work independently
Must be able to operate lawn maintenance equipment such as lawnmowers, string trimmers, leaf blowers, hedge trimmers, etc.
Able to physically stand, bend, squat, and lift up to 40 pounds"""



docs = [job_2.lower(), job_3.lower()]
j1 = nltk.sent_tokenize(job_1.lower())
j1 = data_processor.clean(j1)[0]
# print (j1)
# exit()
save_docs = ["j1", "j2", "Sales"]

# job_1 = nltk.sent_tokenize(job_1)
# print (job_1)
# print (docs[0])

docs[0] = data_processor.clean([docs[0]])[0]
# print (docs[0])
docs[1] = data_processor.clean([docs[1]])[0]
# docs[2] = data_processor.clean(docs[2])[0]


# exit()
## Split

docs[0] = [d.split(" ") for d in docs[0]]
docs[1] = [d.split(" ") for d in docs[1]]
# docs[2] = [d.split(" ") for d in docs[2]]

nd0 = []
for i in range(0, len(j1) - 5):
    nd0.append(j1[i:i + 5])

j1 = nd0
print (j1)
j1 = [[d.split(" ") for d in a] for a in j1]
# print (j1)
j1 = [[[data_processor.subword(w, w2v) for w in d] for d in a] for a in j1]

j1 = [[pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in a] for a in j1]
# print (j1)
# print (np.array(j1).shape)
rets = []
for z in j1:
    for b in z:
        x = word2vec.predict(b)
        for i in range(len(x)):
            a = x[i]
            a_d = np.linalg.norm(a)
            a = a / a_d
            x[i] = a

        rets.append(sum(x)/len(x))
# print (rets)
# exit()


## Create subwords

docs[0] = [[data_processor.subword(w, w2v) for w in d] for d in docs[0]]
docs[1] = [[data_processor.subword(w, w2v) for w in d] for d in docs[1]]
sd = docs

## Pad

docs[0] = [pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in docs[0]]
# print (docs[0])
docs[1] = [pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in docs[1]]
# docs[1] = pad_sequences(docs[1], maxlen=25, value=len(w2v), padding="post")
# print (docs[1])
# docs[2] = pad_sequences(docs[2], maxlen=25, value=len(w2v), padding="post")
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

sims = []
for i in rets:
    # print (i)
    sims.append(calculate_similarity(i, vecs[0]))

sims2 = []
for i in rets:
    sims2.append(calculate_similarity(i, vecs[1]))

# sims = max(sims)


# print (1/np.linalg.norm(vecs[0]-vecs[1]))
# print (1/np.linalg.norm(vecs[1]-vecs[2]))
# print (1/np.linalg.norm(vecs[2]-vecs[0]))

# print(vecs[0])
# print(vecs[2])

## Sim

# sim = [0, 0, 0]

# sim[0] = calculate_similarity(vecs[0], vecs[1])
# sim[1] = calculate_similarity(vecs[1], vecs[2])
# sim[2] = calculate_similarity(vecs[2], vecs[0])

# ## Display
def get_max_n(n, sims):
    sims_ = sorted(sims)[::-1]
    sims_ = sims_[:n]
    return (sum(sims_)/len(sims_))
    # exit()

def display(sims):
    n = int(len(sims)*0.15)
    n_avg = get_max_n(n, sims)
    n_avg_conv = 1.011951 + (0.0001021114 - 1.011951)/(1 + (n_avg/0.5718397)**5.644143)

    print (len(sims))
    print ("Max Score: %s" % max(sims))
    # print ("Converted Max Score: %s"  % (1.003382 + (0.00004280836 - 1.003382)/(1 + (max(sims)/0.7101592)**14.41005)))
    print ("Average Score: %s" % (sum(sims)/len(sims)))
    print ("%s Average Score: %s" % (n, n_avg))
    print ("%s Converted Average Score: %s" % (n, n_avg_conv))




# print ("%s | %s: %s" % (save_docs[0], save_docs[1], sim[0]))
# print ("%s | %s: %s" % (save_docs[1], save_docs[2], sim[1]))
# print ("%s | %s: %s" % (save_docs[2], save_docs[0], sim[2]))
print (" ")
display(sims)
print (" ")
display(sims2)


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