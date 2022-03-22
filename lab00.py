import sys

sys.path.append("../")
from utils import *

zeros_2d = np.zeros((7, 10))

zeros_2d[::2, 1::2] = 1
zeros_2d[1::2, ::2] = 1


# print(zeros_2d)
# print(ones_2d)
# print(zeros_2d)

def create_cartesian_product(vec1, vec2):
    def cartesian_product(vec1, vec2):
        # np.repeat([1, 2, 3], 4) -> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        # np.tile([1, 2, 3], 4)   -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        return np.transpose(np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))]))


def find_closest(a, n):
    a = np.array(a)
    closest = np.argmin(np.abs(a - n))  # index of smallest element
    return a[closest]


# find_closest([1,4,5,2],2)

def check_dependencies(matrix_):
    pass


# def create_flight_df(cities_poss, nrows=100):
#     flight_df = pd.DataFrame({"departure": np.array(np.random.choice(cities_poss,size=nrows)),
#                               "destination" : np.array(np.random.choice(cities_poss,size=nrows)),
#                              "price": np.array(np.random.randint(100, 400,size=nrows))})
#     print(flight_df.head())
# #,columns=["city of departure", "city of destination", "price"]
#
# #individual_df = pd.DataFrame(np.array([np.random.randint(2000000, 3000000,50), np.random.uniform(1.50, 1.70, size = 50), np.random.uniform(45, 90, size = 50)]).transpose(), columns=['ID','Height','Weight'])
#
# create_flight_df(["Beijing", "Moscow", "New-York", "Tokyo", "Paris", "Cairo", "Santiago", "Lima", "Kinshasa", "Singapore",
#           "New-Delhi", "London", "Ankara", "Nairobi", "Ottawa", "Seoul", "Tehran", "Guatemala", "Caracas", "Vienna"])


# def create_flight_df(cities, nrows=20):
#     df = pd.DataFrame([], columns=["Departure", "Destination", "Price"])
#
#     while df.shape[0] < nrows:
#         dep, dest = np.random.choice(cities, size=2, replace=False)
#         price = np.random.randint(100, 400)
#
#         if not ((df["Departure"] == dep) & (df["Destination"] == dest)).any():
#             df = df.append({"Departure": dep, "Destination": dest, "Price": price}, ignore_index=True)
#     return df
#
#
# cities = ["Beijing", "Moscow"]
#
# flights = create_flight_df(cities)
# flights.head()

students_df = pd.read_csv("../datasets/Students_Performance.csv")
students_df.head()

df_count_ethnicities = students_df.groupby(['race.ethnicity']).size().reset_index(name='Count')
print(df_count_ethnicities)