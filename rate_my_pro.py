
import RateMyProfessor_Database_APIs as rmp

prof = rmp.fetch_a_professor(2633588)
print(type(prof))
print(prof)