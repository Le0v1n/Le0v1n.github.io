class People:
    def __init__(self, name, age):
        self.name = name;
        self.age = age;

    def dis_name(self):
        print("name is:", self.name)

    def set_age(self, age):
        self.age = age;

    def dis_age(self):
        print("age is:", self.age);


class Student(People):
    # 必须重新初始化self(实例对象)的属性才可以
    def __init__(self, name, age, school_name):
        self.name = name;
        self.age = age;
        self.school_name = school_name;

    def dis_student(self):
        print("school name is:", self.school_name);


# 实例化对象
student = Student("Leovin", "23", "North Minzu University");
student.dis_student();  # 调用自身的方法
student.dis_name();  # 调用父类的方法
student.dis_age();  # 调用父类的方法
student.set_age(25);  # 调用父类的方法
student.dis_age();  # 调用父类的方法

"""
Res:
    school name is: North Minzu University
    name is: Leovin
    age is: 23
    age is: 25
"""
