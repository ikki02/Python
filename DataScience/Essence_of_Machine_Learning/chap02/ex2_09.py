class Person:
    def __init__(self, first_name="", last_name=""):
        self.first_name = first_name
        self.last_name = last_name


person1 = Person("John", "Smith")
print(person1.first_name, person1.last_name)
person2 = Person()
person2.first_name = "Robert"
person2.last_name = "Johnson"
print(person2.first_name, person2.last_name)
