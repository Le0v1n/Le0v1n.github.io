class Wangjp:
    def replaceSpace(self, s):
        A = list(s)
        for i in A:
            if i == ' ':
                i = '20%'
