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

# import nltk
from nameparser.parser import HumanName
from nltk.corpus import wordnet

# person_names=person_list
person_list = []
def get_human_names(text):
    print ("starting")
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)

    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []
    return person_list
#     print (person_list)

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
# 
job_1 = """Obinna Okonkwo. 9108 Jousting Lane, UPPER MARLBORO, Maryland, 20772 * 2405849193 * Obinna 2013@gmail.com. WORK EXPERIENCE. HOWARD UNIVERSITY, Washington, District of Columbia Operations Coordinator, Aug 2015 - Aug 2018. Work with internal team to ensure all visitors are receiving the service we expect. Handle equipment allocations, maintenance, inventory, and service timelines for all the Center's owned equipment.. Balance the needs of our customers with the needs of our operators; make the customer feel appreciated and heard while ensuring that internal operations are at the forefront of every decision.. T.A.G LABS INC, Washington, DC, District of Columbia Student Intern - Game Design Instructor, Mar 2017 - Dec 2017 . Provide a Safe and Fun Learning Environment for Students with the DC Public School system. Provide. students with meaningful background in programs such as Python, C  , Javascript, and Phasor. Contribute to the achievement of the organization's goals through instruction and commitment to T.A.G Labs Inc's the school's Principles. Student Intern - T.A.G Business Analyst, Jan 2018 - Aug 2018. . Strong attention to detail, highly organized and able to thrive in a busy, deadline driven atmosphere. Experience working with business users to gather requirements, writing functional and technical specifications and communicating technical requirements. Ability to grasp technical concepts and communicate to developers as well as communicate technical information to a non-technical audience.. Strong analytical skills, work ethic, independence, problem solving ability, and positive attitude. ENVIRONMENTAL HEALTH ENGINEERING SERVICES, Upper Marlboro, Maryland Industrial Hygenist, Jun 2014 - May 2018. . Operate and maintain instruments including sampling pumps, calibrators, noise dosimeters, sound level meters, and direct reading gas meters. Develop sampling strategies for industrial hygiene projects. Collect notes on activities performed by workers being monitored and/or maintain daily log of activities related to project work." Complete data sheets, chain of custody forms, and prepare samples for shipment to laboratory.. EDUCATION OXON HILL HIGH SCHOOL, Oxon Hill, Maryland High School Diploma, May 2013. HOWARD UNIVERSITY, Washington, DC, Maryland B.S in Electrical Engineering , May 2018. TOWSON UNIVERSITY, Towson, Maryland M.S in Computer Science Candidate, Expected graduation, May 2020. ADDITIONAL SKILLS . Python | C   | Microsoft Office Suite | Matlab | Agile Environment . Network  Certification *In Progress*."""
job_1 = """Tucker Siegel Co-founder and Chief Technology Officer. Tucker Siegel 266 E Dudley Avenue Westfield, NJ 07090. 908.419.9415 tuckers@krow.network siegel.tucker@gmail.com. Skills. Full stack developer with experience in languages ranging from Javascript to Python, as well as deep learning and artificial intelligence algorithms. Extensive experience in customer service and customer relations in a retail and services setting. Experience. Krow Network / Co-founder and Chief Technology Officer August 2018 - PRESENT, NEWARK, NJ Leading multi-platform development to build a useable product for job seekers in the United States. Oversee all development of user facing and back-end services, and fully operate all artificial intelligence programs. Best Buy / Connected Devices Sales Consultant OCTOBER 2018 - PRESENT, UNION, NJ Develop relationships with customers to understand their situation and recommend the right solution. Implement sales tactics as well as sell recurring services and sign ups for the store credit card. Westfield, NJ / Intern at Technology Advisory Committee OCTOBER 2018 - PRESENT, WESTFIELD, NJ Meet with team members once a month to develop new technological solutions to current issues facing the town. Advise the mayor on budgeting concerns, and implementation of new ideas. Gold Medal Fitness / Member Services Representative MAY 2017 - PRESENT, GARWOOD, NJ Serve at the front desk for Gold Medal Fitness at the Garwood and Cranford, NJ locations. Responsible for handling member complaints, snack and beverage sales, and enrolling new members."""



# job_1 = """CIDNEY M. KING. 15703 Swanscombe Loop, Upper Marlboro, Maryland, 20774. cidneyking@ifly-youth.org. (202) 738-6261. PROFESSIONAL EXPERIENCE. Founder & President. May 2017 Present iFLY Youth Inc. - Washington, D.C. Created iFLY Youth, an organization focused on providing middle school girls of color from underserved. neighborhoods in the Greater Washington Area the opportunity to travel, serve, and learn internationally. . Secured over $26,000 in seed funding from private foundations, corporations, and individual donors. Work alongside and lead board of directors in managing and implementing all structural, financial, and operational processes and continually improve policies and procedures to further streamline efficient operations. Developed partnership with E.L. Haynes Public Charter School and piloted 6-day international travel experience to Costa Rica for 10 of their students. Manager of Operations & Community Engagement Chair. January 2017 Present Capital Village Schools - Washington, D.C. Collaborate with a team of school leaders, community leaders, professionals and educators in the design and launch of a network of micro-schools that will prepare all students, regardless of background or circumstance, to reach their full potential in college, career and life in order to become agents of change in their communities Facilitate strategic partnerships that will aid in building small learning communities focused on relevant realworld problems, personalized to each student's individual needs, and centered around the whole child Plan and execute community engagement events, including overseeing the budget, purchasing materials and onsite logistics Finalized student recruitment, enrollment and parent involvement plans for 4 week-summer pilot of Capital Village's school model. Manager of Recruitment. July 2016 May 2017 Teach for America  Washington, D.C. Identified, cultivated, and influenced 19 diverse top-talent seniors from University of Maryland College Park to apply to and join Teach For America through 1:1 meetings, ongoing phone/email outreach, classroom and student organization presentations, information sessions, campus events, social media, and faculty networking Managed a team of 4 part-time student interns to reach rigorous weekly goals, create innovative events and marketing strategy, support pipeline development, and coach on how to build awareness of Teach For America and the issue of educational inequity on campus Utilized Salesforce to inform strategic decisions and track campaign actions, including pipeline development, event planning, relationship building, and confirmation. Mathematics Teacher. July 2014 July 2016 Al Badiya School -Al Wagon, United Arab Emirates . Scaffolded large learning tasks into meaningful and manageable segments for English second language. learners in 6th grade to ensure 100 percent of students achieved their individual growth goals . Managed systems for tracking large amounts of data geared toward reaching an overall class goal of at least. 80 percent mastery of all math objectives Observed teammates bi-monthly and offered constructive criticism for best practices to increase math literacy and heighten student ownership of learning. English Language Arts Teacher. May 2012 July 2014 Walipp Preparatory Academy - Houston, Texas . Launched Leadership Atlanta Program, which provided 12 girls with the opportunity to travel to. colleges/universities in Atlanta, Georgia and spearheaded 12-thousand-dollar fundraising goal Created culturally relevant lesson plans for 130 students in grades 6th through 8th to ensure 100 percent of learners achieved a minimum of 75 percent mastery of each reading objective."""
# job_1 = """NURSING TEMPLATE. 500 Resume Sample Road Atlanta, Georgia (123) 123-1234 sample.nurse resume.com. Summary. Awarded Northside Hospital, Nurse of the Year. Awarded Atlanta Medical Center, Preceptor of the Year. Dedicated, compassionate, knowledgeable Registered Nurse with over 13 years of experience, 10 of those years in the critical care setting. Meticulously detail-oriented, observant and proactive. Performs well in high pressure, fast paced environment. Skilled in working with doctors for efficient patient management and a team player who is able to relate well with coworkers Special practice in the cardiac cath/EP labs and post op recovery of cath/EP patients. Experience in TAVR surgeries. Experience in Preop / PACU areas specializing in ENT, eye. orthopedic, and GYN surgeries. Education. Georgia State University, Atlanta, Georgia BSc. Nursing Graduated - May 2004. Employment History. Surgecenter of Louisville, Louisville, Kentucky Preop Nurse /PACU Nurse January 2017 - Present I am currently working in an outpatient surgery center in both the Preop and the Post Anesthesia Care Unit (PACU) departments. We have four operating rooms and specialize in eye, ENT, orthopedic, and gynecological surgeries. I work closely with the center's physicians and anesthesiologists. I care for infants, children and adults. Atlanta Medical Center, Atlanta, Georgia Registered Nurse / Staff Nurse October 2014 September 2017 I worked in a fast paced 5 room cath lab. We performed up to 30 cases daily which varied from diagnostic caths to ablations, to peripheral cases, to Anari procedures. I was also a part of Atlanta Medical Center's STEMI team. Atlanta Medical Center is one of the few hospitals in Georgia to receive accreditation as a heart attack receiving closely with both physicians and my coworkers to care for my patients to the best of my ability. I was also on the Quality council and worked closely with Unit Based Shared Governance."""
# job_1 = """Shawnee M. Johnson smjohnson3195@gmail.com. (240) 715-5853. www.linkedin.com/in/shawneemjohnson. EDUCATION: B. A. Sociology, University of Maryland, College Park (UMD) Relevant Courses: Social Psychology, Research Methods in Sociology, Social Stratification & Inequality, Writing For Non-Profits and Communication Core Competencies: Data entry & analysis, administrative support, excellent writing and communication skills and strong interpersonal skills Skills & Abilities: Advanced proficiency in Microsoft Office applications, Intermediate proficiency in SPSS & GSS, and novice comprehension of French & Spanish. EXPERIENCE Careers in Nonprofits | Administrative Assistant (Temporary) | Washington, D.C. | April 2018 - Present. . Schedules and coordinates meetings for executives and facilities . Sorts and distributes incoming communication data, including faxes, letters and emails " Responds to phone calls, walk ins, visitors and written requests for information " Maintains invoice logs and ensures invoice accuracy. . Creates spreadsheets and presentations for executives TJ Maxx | Sales-Cash Office Associate | Greenbelt, MD | July 2012  April 2018. . Ensured customer satisfaction . Handled money transports, change provision for registers, daily account reconciliation and payroll verifications " Intercepted phone calls in a professional manner providing answers and transferring calls when needed. . Trained new employees for cashiering and merchandising The Petey Greene Program | Tutor | Jessup, MD | September 2017 - December 2017. . Provided free quality tutoring to incarcerated people in preparation for GED testing & test taking strategies . Attended workshops focused on cultural humility, adult education, and criminal justice " Designed comprehensive worksheets for knowledge assessments. Imparted content area instruction for small groups or with individual students Northwestern University Summer Research | Scholar | Evanston, IL | June 2017 - August 2017. . Organized independent research on minority population in conjunction to social class and immigrant status . Interpreted and analyzed quantitative and qualitative results through Microsoft Office & SPSS. Presented PowerPoint research for Northwestern University Research Symposium to academics and general. public UMD | Undergraduate Research Assistant College Park, MD | August 2016  January 2017. . Arranged interview meetings for staff and faculty research participants through email correspondence . Compiled research literature in preparation for reports focused on the impact of disability on students' lives. . Advertised and attended student focus group workshops centered around disability and allyship EveryMind | Intern | Rockville, MDL June 2015  September 2015. . Consolidated hard copy data to electronic data using Excel program . Managed paperwork and updated client information in company database " Accomplished administrative tasks such as maintaining documents and corresponding messages. ACHIEVEMENTS. . Alpha Kappa Delta Society Member . Ronald E. McNair Post-Baccalaureate Achievement Program Scholar."""
names = get_human_names(job_1)
print (names)
# exit()
name = names[0]
print (name)
parse = CommonRegex(job_2)
print (parse)
# print (parse.names)
# exit()
for i in parse.phones:
    job_2 = job_2.replace(i, "")

for i in parse.emails:
    job_2 = job_2.replace(i, "")

for i in parse.street_addresses:
    job_2 = job_2.replace(i, "")


job_3 = """**A-Line Staffing** is seeking a qualified candidate for the position of **UI
Engineer** located in **Denver, CO**

Please review the desired qualifications listed below and apply or contact
**Nate Namiotka** with questions.

The Role The UI Developer will play a key role in the development and
maintenance of commercial software product(s) in building reliable, testable,
scalable and high-performing enterprise class web applications for our global
Fortune 500 customers. The successful candidate must be able to design,
develop and automate the testing of responsive, front-end user interfaces
across multiple browsers. You will work with all software product development
disciplines (including UX, engineering, architecture, product owners, testers
and analysts) to develop and maintain solutions that meet the needs of the
business, utilizing industry standard best practices and modern software
engineering techniques.

**Responsibilities include:**

· Design, code, test and debug UI enhancements for commercial software
product(s)

· Partner with our UX team in the implementation of complex UI designs

· Drive maintainable and reusable solutions that are both secure and adhering
to acceptable performance benchmarks

· Embrace, implement and evangelize test automation as part of the fabric of
the team

· Conduct code reviews for peers and/or junior team members

· Participle in the evaluation of emerging technology and tools

 ****

**Requirements:**

· Bachelor's degree and 4+ years of relevant hands-on experience in full User
Interface development.

· Must understand and have practiced UCD (user-centered design) concepts

· Experience with the latest ASP.NET 5, MVC and Web Pages

· Applied knowledge of HTML5, CSS3, Java Script, Angular 2+, Data Tables,
TypeScript and JQuery is required

· Proficiency with code optimization and debugging technics with JavaScript,
HTML and third-party libraries such as Angular, JQuery, etc.

· Experience in responsive web application development (RWD) is required.

· Excellent applied knowledge in writing cross-browser compatible code (IE,
Chrome, Firefox, Safari).

· Must be a creative and independent thinker who can work effectively in a
team environment.

· Experience implementing a well architected, componentized front-end system.

· Experience with unit and system testing frameworks and techniques (e.g.
Jasmine)

· Experience with Software as a Service (SaaS), multi-tenancy application
development

· Experience with browser-based web applications delivered over the Internet
that support high-volume user concurrency

· Experience using Agile Scrum software development life cycle is a MUST.

· Experience using source control management tools such as Microsoft Team
Foundation Server (TFS)

· Excellent organizational skills, multitasking ability and proven analytical,
methodical thinking, problem solving and decision-making skills

· Excellent verbal and writing skills, including the ability to research,
design, and write new documentation, as well as to maintain and improve
existing material

· Experience working for a national or global commercial software product
company requiring software product usage and high availability in a 24/7/365
model.

· Experience working with globally distributed business stakeholders and
engineering teams

· Experience working with collaboration and issue tracking tools such as
Confluence and JIRA Experience working in a large, fast-paced project
environment

· Strong work ethic is a must

· Proactive: Must be willing to seek out information and solutions that are
pertinent to their responsibilities and key to their delivery

· Must be willing to remediate code written by others

· Team player attitude with a focus on the success of the team is a MUST

· Must be able to work in a structured development environment and follow
coding standards and design patterns.

· Proactively collaborates with other teams/team members (UX, Enterprise
Architecture, Quality Assurance, Product Owners, Business Analysts,
Information Security, regional teams) across global time zones with the upmost
professionalism, tact and resolve

· Experience with Test Driven Development or Behavior Driven Development is
preferred.

· Experience developing applications requiring internationalization and
localization for global markets is preferred.

**I look forward to hearing from you soon!**

**Nate Namiotka – A-Line Staffing**"""



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
print (len(j1))
for i in range(0, len(j1)-5):
    nd0.append(j1[i:i+5])
    # nd0.append([j1[i]])

j1 = nd0
# print (np.array(j1).shape)
j1 = [[d.split(" ") for d in a] for a in j1]
# print (j1)
j1 = [[[data_processor.subword(w, w2v) for w in d] for d in a] for a in j1]

j1 = [[pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in a] for a in j1]
# print (j1)
# print (np.array(j1).shape)
rets = []
for z in j1:
    # print (np.array(z[0]).shape)
    x = []
    for b in z:
        e = word2vec.predict(b)
        
        for i in range(len(e)):
            a = e[i]
            a_d = np.linalg.norm(a)
            a = a / a_d
            e[i] = a
        x.extend(e)
    if (len(x) != 0):
        # print (sum(x))
        # print (len(x))
        rets.append(sum(x)/len(x))

# rets = sum(rets)/len(rets)
# rets = [rets]
    # rets.append(sum(x)/len(x))

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
# print (len(rets))
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
    sims_ = sims_[:-1]
    print (sims)
    return (sum(sims_)/len(sims_))
    # exit()

def display(sims):
    # print (len(sims))
    # print (len(nd0))
    sims = sorted(sims)[::-1]
    # sims = sims[:-1]
    print (sims)
    n = int(len(sims)*.5)
    # for a, e in zip(sims, nd0):
    #     print (a, e)
    n_avg = sum(sims)/len(sims)
    n_avg_conv = 1.007485 + (0.5987585 - 1.007485)/(1 + (max(sims)/0.7986861)**17.69422)

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
display(sims2)
print (" ")
# display(sims2)


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