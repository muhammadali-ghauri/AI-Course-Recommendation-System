from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("=================================================")
print("\tAI Course Recommendation System")
print("=================================================")

print("\nWelcome!")
print("This system recommends courses based on your interests.")


# Course dataset
courses = [
    {
        "name": "Python for Beginners",
        "description": "python programming basics coding variables loops functions"
    },
    {
        "name": "Artificial Intelligence Fundamentals",
        "description": "artificial intelligence machine learning python algorithms intelligent systems"
    },
    {
        "name": "Web Development",
        "description": "html css javascript frontend web development websites"
    },
    {
        "name": "Data Science with Python",
        "description": "python data science pandas numpy data analysis visualization"
    },
    {
        "name": "Machine Learning Basics",
        "description": "machine learning python models algorithms data prediction"
    },
    {
        "name": "Deep Learning Fundamentals",
        "description": "deep learning neural networks python artificial intelligence models"
    },
    {
        "name": "Database Management with SQL",
        "description": "sql databases mysql queries tables data database management"
    },
    {
        "name": "Java Programming",
        "description": "java programming object oriented programming classes methods coding"
    },
    {
        "name": "Cybersecurity Basics",
        "description": "cybersecurity network security threats privacy protection systems"
    },
    {
        "name": "Data Structures and Algorithms",
        "description": "data structures algorithms programming arrays linked lists sorting searching"
    }
]


# Take user's interests
print("\nTell me what you are interested in.")
user_interest = input("Enter your interests: ").lower().strip()


# Extract course descriptions
course_descriptions = []

for course in courses:
    course_descriptions.append(course["description"])


# Convert course descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer()

course_vectors = vectorizer.fit_transform(course_descriptions)


# Convert user interests into the same vector space
user_vector = vectorizer.transform([user_interest])


# Calculate similarity between the user and each course
similarity_scores = cosine_similarity(user_vector, course_vectors)
scores = similarity_scores[0]


# Pair each course with its similarity score
recommendations = []

for i in range(len(courses)):
    recommendations.append((scores[i], courses[i]["name"]))


# Rank courses from highest similarity to lowest
recommendations.sort(reverse=True)


print("\n========================================")
print("\tTop Course Recommendations")
print("========================================")

found_match = False

# Display the top 3 matching courses
for score, course_name in recommendations[:3]:
    if score > 0:
        percentage = score * 100
        print(f"{course_name} - {percentage:.1f}% Match")
        found_match = True


# Handle cases where no course matches the user's interests
if not found_match:
    print("Sorry, no matching courses were found.")

print("========================================")