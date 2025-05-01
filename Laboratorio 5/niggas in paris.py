from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

data = {
    
    # Conjuntos
    # R: Conjunto de recursos humanitarios disponibles para distribución
    # A: Conjunto de aviones disponibles para realizar los envíos
    # V: Conjunto de viajes posibles para cada avión
    # Z: Conjunto de zonas afectadas que requieren ayuda humanitaria
    
    'R': ['Alimentos', 'Medicinas', 'EquiposMédicos', 'Agua', 'Mantas'],
    'A': [1, 2, 3, 4],
    'V': [1, 2],
    'Z': ['A', 'B', 'C', 'D'],


    # Valor de impacto
    'valor_impacto': {'Alimentos': 50,
            'Medicinas': 100,
            'EquiposMédicos': 120,
            'Agua': 60,
            'Mantas': 40},


    # Peso por unidad
    'peso_unidad': {'Alimentos': 5,
            'Medicinas': 2,
            'EquiposMédicos': 0.3,
            'Agua': 6,
            'Mantas': 3},


    # Volumen por unidad
    'volumen_unidad': {'Alimentos': 3,
            'Medicinas': 1,
            'EquiposMédicos': 0.5,
            'Agua': 4,
            'Mantas': 2},


    # Disponibilidad total
    'disponibilidad': {'Alimentos': 12,
            'Medicinas': 15,
            'EquiposMédicos': 40,
            'Agua': 15,
            'Mantas': 20},


    # Capacidad de peso
    'capacidad_peso_avion': {1: 40,
              2: 50,
              3: 60,
              4: 45},


    # Capacidad de volumen
    'capacidad_volumen_avion': {1: 35,
              2: 40,
              3: 45,
              4: 38},


    # Costo fijo
    'costo_fijo':   {1: 15,
              2: 20,
              3: 25,
              4: 18},


    # Costo variable
    'costo_variable': {1: 0.020,
              2: 0.025,
              3: 0.030,
              4: 0.022},

    # Zonas
    'distancia': {'A': 800,
               'B': 1200,
               'C': 1500,
               'D': 900},

    # Multiplicador de impacto
    'multiplicador':    {'A': 1.2,
               'B': 1.5,
               'C': 1.8,
               'D': 1.4},

    # Necesidades mínimas por zona
    'necesidades_min': {
        ('Alimentos', 'A'): 8,  ('Alimentos', 'B'): 12,  ('Alimentos', 'C'): 16,  ('Alimentos', 'D'): 10,
        ('Medicinas', 'A'): 2,  ('Medicinas', 'B'): 3,   ('Medicinas', 'C'): 4,   ('Medicinas', 'D'): 2,
        ('EquiposMédicos', 'A'): 0.6, ('EquiposMédicos', 'B'): 0.9, ('EquiposMédicos', 'C'): 1.2, ('EquiposMédicos', 'D'): 0.6,
        ('Agua', 'A'): 6,       ('Agua', 'B'): 9,       ('Agua', 'C'): 12,      ('Agua', 'D'): 8,
        ('Mantas', 'A'): 3,     ('Mantas', 'B'): 5,     ('Mantas', 'C'): 7,     ('Mantas', 'D'): 4,
    }
}


def build_model(data):
    model = ConcreteModel(data)

    model.R = Set(initialize=data['R'])
    model.A = Set(initialize=data['A'])
    model.Z = Set(initialize=data['Z'])
    model.V = Set(initialize=data['V'])

    model.valor_impacto = Param(model.R, initialize=data['valor_impacto'])
    
    model.peso_unidad = Param(model.R, initialize=data['peso_unidad'])
    
    model.volumen_unidad = Param(model.R, initialize=data['volumen_unidad'])
    
    model.disponibilidad = Param(model.R, initialize=data['disponibilidad'])
    
    model.capp_a = Param(model.A, initialize=data['capacidad_peso_avion'])
    
    model.capv_a = Param(model.A, initialize=data['capacidad_volumen_avion'])
    
    model.costo_fijo = Param(model.A, initialize=data['costo_fijo'])
    
    model.cvar = Param(model.A, initialize=data['costo_variable'])
    
    model.distancia = Param(model.Z, initialize=data['distancia'])
    
    model.multiplicador = Param(model.Z, initialize=data['multiplicador'])
    
    model.necesidades_min = Param(model.R, model.Z, initialize=data['necesidades_min'], default=0)


    model.X      = Var(model.R, model.A, model.V, model.Z, domain=NonNegativeReals)

    model.Y      = Var(model.A, domain=Binary)

    model.asignar_avz  = Var(model.A, model.V, model.Z, domain=Binary)

    model.y_aux  = Var(model.A, model.V, domain=Binary)


    def capacidad_peso_rule(m, a, v, z):
        return sum(m.X[r,a,v,z]*m.peso_unidad[r] for r in m.R) <= m.capp_a[a]
    model.capacidad_peso_rule = Constraint(model.A, model.V, model.Z, rule=capacidad_peso_rule)


    def capacidad_volumen_rule(m, a, v, z):
        return sum(m.X[r,a,v,z]*m.volumen_unidad[r] for r in m.R) <= m.capv_a[a]
    model.capacidad_volumen_rule = Constraint(model.A, model.V, model.Z, rule=capacidad_volumen_rule)


    def disponibilidad_total_rule(m, r):
        return sum(m.X[r,a,v,z] for a in m.A for v in m.V for z in m.Z) <= m.disponibilidad[r]
    model.disponibilidad_total_rule = Constraint(model.R, rule=disponibilidad_total_rule)

    
    def necesidades_min_rule(m, r, z):
        return sum(m.X[r,a,v,z] for a in m.A for v in m.V) * m.peso_unidad[r] >= m.necesidades_min[r,z]
    model.necesidades_min_rule = Constraint(model.R, model.Z, rule=necesidades_min_rule)

    
    def unica_zona_rule(m, a, v):
        return sum(m.asignar_avz[a,v,z] for z in m.Z) <= 1
    model.unica_zona_rule = Constraint(model.A, model.V, rule=unica_zona_rule)

   
    def activar_avion_rule(m, a):
        return m.Y[a]*len(m.V)*len(m.Z) >= sum(m.asignar_avz[a,v,z] for v in m.V for z in m.Z)
    model.activar_avion_rule = Constraint(model.A, rule=activar_avion_rule)

    
    def medicinas_no_avion_1_rule(m, v, z):
        return m.X['Medicinas',1,v,z] == 0
    model.medicinas_no_avion_1_rule = Constraint(model.V, model.Z, rule=medicinas_no_avion_1_rule)

    
    M_big = sum(data['disponibilidad'][r]*data['peso_unidad'][r] for r in data['R'])
    def agua_ok(m, a, v):
        return sum(m.X['Agua',a,v,z] for z in m.Z) <= M_big * m.y_aux[a,v]
    def equipos_ok(m, a, v):
        return sum(m.X['EquiposMédicos',a,v,z] for z in m.Z) <= M_big * (1-m.y_aux[a,v])
    model.agua_ok    = Constraint(model.A, model.V, rule=agua_ok)
    model.equipos_ok = Constraint(model.A, model.V, rule=equipos_ok)

    
    def enlace_asignacion_rule(m, r, a, v, z):
        return m.X[r,a,v,z] <= M_big * m.asignar_avz[a,v,z]
    model.enlace_asignacion_rule = Constraint(model.R, model.A, model.V, model.Z, rule=enlace_asignacion_rule)

    return model


solver = SolverFactory('glpk')


m1 = build_model(data)
m1.obj = Objective(
    expr=sum(m1.valor_impacto[r]*m1.X[r,a,v,z]*m1.multiplicador[z]
             for r in m1.R for a in m1.A for v in m1.V for z in m1.Z),
    sense=maximize)
solver.solve(m1)
max_impacto = value(m1.obj)


m2 = build_model(data)
m2.obj = Objective(
    expr=sum(m2.costo_fijo[a]*m2.Y[a] for a in m2.A)
       + sum(m2.cvar[a]*m2.asignar_avz[a,v,z]*m2.distancia[z]
             for a in m2.A for v in m2.V for z in m2.Z),
    sense=minimize)
solver.solve(m2)
min_costo = value(m2.obj)


m3 = build_model(data)
m3.obj = Objective(
    expr=sum(m3.costo_fijo[a]*m3.Y[a] for a in m3.A)
       + sum(m3.cvar[a]*m3.asignar_avz[a,v,z]*m3.distancia[z]
             for a in m3.A for v in m3.V for z in m3.Z),
    sense=maximize)
solver.solve(m3)
max_costo = value(m3.obj)



alpha_values = np.linspace(0.01, 0.99, 19)
pareto_points = []

for alpha in alpha_values:
    m = build_model(data)

    Z1 = sum(m.valor_impacto[r]*m.X[r,a,v,z]*m.multiplicador[z]
             for r in m.R for a in m.A for v in m.V for z in m.Z)
    Z2 = (sum(m.costo_fijo[a]*m.Y[a] for a in m.A)
         + sum(m.cvar[a]*m.asignar_avz[a,v,z]*m.distancia[z]
               for a in m.A for v in m.V for z in m.Z))

    Z1_n = Z1 / max_impacto
    Z2_n = Z2 / max_costo
    m.obj = Objective(expr=alpha*Z1_n - (1-alpha)*Z2_n,
                      sense=maximize)

    result = solver.solve(m)

    if result.solver.status == SolverStatus.ok:
        impacto_val = value(Z1)
        costo_val   = value(Z2)
        pareto_points.append((impacto_val, costo_val))


rows = []
for z in m.Z:
    for r in m.R:
        needed = data['necesidades_min'][(r, z)]
        delivered = sum(
            value(m.X[r, a, v, z]) * data['peso_unidad'][r]
            for a in m.A for v in m.V
        )
        rows.append({
            'Zona':      z,
            'Recurso':   r,
            'Necesitaba': needed,
            'Entregó':   round(delivered, 3)
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))


xs = [p[1] for p in pareto_points]
ys = [p[0] for p in pareto_points]
plt.plot(xs, ys, marker='o')
plt.xlabel('Costo (miles USD)')
plt.ylabel('Impacto social (miles USD)')
plt.title('Frente de Pareto Aproximado')
plt.grid(True)
plt.show()