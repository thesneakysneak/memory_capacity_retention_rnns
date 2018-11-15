# from pyparsing import LineEnd, oneOf, Word, nums, Combine, restOfLine, \
#     alphanums, Suppress, empty, originalTextFor, OneOrMore, alphas, \
#     Group, ZeroOrMore
#
# data = """\
# . 55 MORILLO ZONE VIII,
# BARANGAY ZONE VIII
# (POB.), LUISIANA, LAGROS
# F
# 01/16/1952
# ALOMO, TERESITA CABALLES
# 3412-00000-A1652TCA2
# 12
# . 22 FABRICANTE ST. ZONE
# VIII LUISIANA LAGROS,
# BARANGAY ZONE VIII
# (POB.), LUISIANA, LAGROS
# M
# 10/14/1967
# AMURAO, CALIXTO MANALO13
# """
#
# NL = LineEnd().suppress()
# gender = oneOf("M F")
# integer = Word(nums)
# date = Combine(integer + '/' + integer + '/' + integer)
#
# # define the simple line definitions
# gender_line = gender("sex") + NL
# dob_line = date("DOB") + NL
# name_line = restOfLine("name") + NL
# id_line = Word(alphanums+"-")("ID") + NL
# recnum_line = integer("recnum") + NL
#
# # define forms of address lines
# first_addr_line = Suppress('.') + empty + restOfLine + NL
# # a subsequent address line is any line that is not a gender definition
# subsq_addr_line = ~(gender_line) + restOfLine + NL
#
# # a line with a name and a recnum combined, if there is no ID
# name_recnum_line = originalTextFor(OneOrMore(Word(alphas+',')))("name") + \
#     integer("recnum") + NL
#
# # defining the form of an overall record, either with or without an ID
# record = Group((first_addr_line + ZeroOrMore(subsq_addr_line))("address") +
#     gender_line +
#     dob_line +
#     ((name_line +
#         id_line +
#         recnum_line) |
#       name_recnum_line))
#
# # parse data
# records = OneOrMore(record).parseString(data)
#
# # output the desired results (note that address is actually a list of lines)
# for rec in records:
#     if rec.ID:
#         print("%(name)s, %(ID)s, %(address)s, %(sex)s, %(DOB)s" % rec)
#     else:
#         print("%(name)s, , %(address)s, %(sex)s, %(DOB)s" % rec)
# print()
#
# # how to access the individual fields of the parsed record
# for rec in records:
#     print(rec.dump())
#     print()