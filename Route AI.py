import streamlit as st
import pandas as pd
import numpy as np
import math
import simpy
import folium
from streamlit_folium import st_folium

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RouteAI",
    page_icon="🚚",
    layout="wide"
)


# =========================================================
# DISTANCE CALCULATION
# =========================================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def distance_matrix(df):

    coords = df[
        ["latitude", "longitude"]
    ].to_numpy(float)

    n = len(df)

    matrix = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            if i != j:

                matrix[i, j] = haversine(
                    coords[i, 0],
                    coords[i, 1],
                    coords[j, 0],
                    coords[j, 1]
                )

    return matrix


def route_distance(route, distance_matrix):

    total = 0

    for i in range(len(route) - 1):

        total += distance_matrix[
            route[i],
            route[i + 1]
        ]

    return total


# =========================================================
# ROUTING ALGORITHM
# =========================================================

def nearest_neighbor(customers, distance_matrix):

    remaining = set(customers)

    route = [0]

    current = 0

    while remaining:

        next_customer = min(
            remaining,
            key=lambda x: distance_matrix[current, x]
        )

        route.append(next_customer)

        remaining.remove(next_customer)

        current = next_customer

    route.append(0)

    return route


def two_opt(route, distance_matrix):

    best = route[:]

    improved = True

    while improved:

        improved = False

        best_distance = route_distance(
            best,
            distance_matrix
        )

        for i in range(
            1,
            len(best) - 2
        ):

            for j in range(
                i + 1,
                len(best) - 1
            ):

                candidate = (
                    best[:i]
                    + best[i:j + 1][::-1]
                    + best[j + 1:]
                )

                candidate_distance = route_distance(
                    candidate,
                    distance_matrix
                )

                if (
                    candidate_distance
                    < best_distance - 1e-9
                ):

                    best = candidate

                    best_distance = candidate_distance

                    improved = True

    return best


# =========================================================
# CAPACITY CONSTRAINT
# =========================================================

def split_capacity_route(
    order,
    df,
    capacity
):

    routes = []

    current_route = [0]

    current_load = 0.0

    for customer_index in order:

        demand = float(
            df.loc[
                customer_index,
                "demand"
            ]
        )

        if demand > capacity:

            raise ValueError(
                f"Demand customer "
                f"{df.loc[customer_index, 'customer']} "
                f"({demand:g}) melebihi "
                f"kapasitas kendaraan "
                f"({capacity:g})."
            )

        if (
            current_load + demand > capacity
            and len(current_route) > 1
        ):

            current_route.append(0)

            routes.append(current_route)

            current_route = [0]

            current_load = 0.0

        current_route.append(
            customer_index
        )

        current_load += demand

    current_route.append(0)

    routes.append(current_route)

    return routes


# =========================================================
# GENERATE THREE SCENARIOS
# =========================================================

def build_scenarios(
    df,
    capacity
):

    distance_matrix_data = distance_matrix(df)

    customers = list(
        range(1, len(df))
    )


    # -----------------------------------------------------
    # SCENARIO 1
    # Shortest Distance
    # -----------------------------------------------------

    base_route = nearest_neighbor(
        customers,
        distance_matrix_data
    )

    optimized_route = two_opt(
        base_route,
        distance_matrix_data
    )

    order_shortest = optimized_route[
        1:-1
    ]

    routes_shortest = split_capacity_route(
        order_shortest,
        df,
        capacity
    )


    # -----------------------------------------------------
    # SCENARIO 2
    # Fast Delivery
    # -----------------------------------------------------

    order_fast = sorted(
        customers,
        key=lambda x:
        distance_matrix_data[0, x]
    )

    routes_fast = split_capacity_route(
        order_fast,
        df,
        capacity
    )


    # -----------------------------------------------------
    # SCENARIO 3
    # Capacity-Aware
    # -----------------------------------------------------

    remaining = set(customers)

    order_capacity = []

    while remaining:

        current_customer = 0

        current_load = 0.0

        while remaining:

            feasible = [
                x
                for x in remaining
                if (
                    current_load
                    + float(
                        df.loc[
                            x,
                            "demand"
                        ]
                    )
                    <= capacity
                )
            ]

            if not feasible:
                break

            next_customer = min(
                feasible,
                key=lambda x: (
                    distance_matrix_data[
                        current_customer,
                        x
                    ],
                    -float(
                        df.loc[
                            x,
                            "demand"
                        ]
                    )
                )
            )

            order_capacity.append(
                next_customer
            )

            remaining.remove(
                next_customer
            )

            current_load += float(
                df.loc[
                    next_customer,
                    "demand"
                ]
            )

            current_customer = (
                next_customer
            )

    routes_capacity = split_capacity_route(
        order_capacity,
        df,
        capacity
    )


    return {

        "Shortest Distance":
            routes_shortest,

        "Fast Delivery":
            routes_fast,

        "Capacity-Aware":
            routes_capacity

    }, distance_matrix_data


# =========================================================
# SIMULATION USING SIMPY
# =========================================================

def simulate_routes(
    routes,
    df,
    distance_matrix_data,
    speed_kmh
):

    results = []

    for route_number, route in enumerate(
        routes,
        start=1
    ):

        env = simpy.Environment()

        state = {

            "distance": 0.0,

            "travel_min": 0.0,

            "service_min": 0.0,

            "customers": 0,

            "load": 0.0

        }


        def vehicle_process():

            for i in range(
                len(route) - 1
            ):

                start = route[i]

                destination = route[i + 1]


                # -----------------------------
                # TRAVEL
                # -----------------------------

                distance = float(
                    distance_matrix_data[
                        start,
                        destination
                    ]
                )

                travel_time = (
                    distance
                    / speed_kmh
                    * 60
                )

                yield env.timeout(
                    travel_time
                )


                state["distance"] += (
                    distance
                )

                state["travel_min"] += (
                    travel_time
                )


                # -----------------------------
                # CUSTOMER SERVICE
                # -----------------------------

                if destination != 0:

                    demand = float(
                        df.loc[
                            destination,
                            "demand"
                        ]
                    )

                    service_time = (
                        5
                        + 0.15 * demand
                    )

                    yield env.timeout(
                        service_time
                    )

                    state[
                        "service_min"
                    ] += service_time

                    state[
                        "customers"
                    ] += 1

                    state[
                        "load"
                    ] += demand


        env.process(
            vehicle_process()
        )

        env.run()


        results.append({

            "route_no":
                route_number,

            "distance_km":
                state["distance"],

            "travel_min":
                state["travel_min"],

            "service_min":
                state["service_min"],

            "total_min":
                (
                    state["travel_min"]
                    + state["service_min"]
                ),

            "customers":
                state["customers"],

            "load":
                state["load"]

        })


    return results


# =========================================================
# PERFORMANCE EVALUATION
# =========================================================

def evaluate_scenario(
    scenario_name,
    routes,
    df,
    distance_matrix_data,
    speed,
    capacity
):

    route_results = simulate_routes(
        routes,
        df,
        distance_matrix_data,
        speed
    )


    total_distance = sum(
        item["distance_km"]
        for item in route_results
    )


    total_time = sum(
        item["total_min"]
        for item in route_results
    )


    total_customers = len(df) - 1


    total_load = float(
        df["demand"]
        .iloc[1:]
        .sum()
    )


    utilization = (
        total_load
        /
        (
            capacity
            * max(
                1,
                len(routes)
            )
        )
        * 100
    )


    throughput = (

        total_customers
        /
        (total_time / 60)

        if total_time > 0
        else 0

    )


    return {

        "Scenario":
            scenario_name,

        "Distance (km)":
            total_distance,

        "Delivery Time (min)":
            total_time,

        "Service Time (min)":
            sum(
                item["service_min"]
                for item in route_results
            ),

        "Routes":
            len(routes),

        "Throughput (cust/h)":
            throughput,

        "Utilization (%)":
            utilization,

        "_route_results":
            route_results

    }


# =========================================================
# MAP
# =========================================================

def make_map(
    df,
    routes
):

    center = [

        df["latitude"].mean(),

        df["longitude"].mean()

    ]


    map_object = folium.Map(
        location=center,
        zoom_start=13
    )


    # DEPOT

    folium.Marker(

        [
            df.loc[
                0,
                "latitude"
            ],

            df.loc[
                0,
                "longitude"
            ]

        ],

        tooltip="Depot",

        popup="Depot",

        icon=folium.Icon(
            color="red",
            icon="home"
        )

    ).add_to(map_object)


    # CUSTOMERS

    for i in range(
        1,
        len(df)
    ):

        folium.Marker(

            [
                df.loc[
                    i,
                    "latitude"
                ],

                df.loc[
                    i,
                    "longitude"
                ]

            ],

            tooltip=str(
                df.loc[
                    i,
                    "customer"
                ]
            ),

            popup=(
                f"{df.loc[i, 'customer']}"
                f" — Demand: "
                f"{df.loc[i, 'demand']}"
            )

        ).add_to(map_object)


    # ROUTES

    for route_number, route in enumerate(
        routes,
        start=1
    ):

        points = [

            [
                df.loc[
                    i,
                    "latitude"
                ],

                df.loc[
                    i,
                    "longitude"
                ]

            ]

            for i in route

        ]


        folium.PolyLine(

            points,

            tooltip=(
                f"Route {route_number}"
            ),

            weight=5

        ).add_to(
            map_object
        )


    return map_object


# =========================================================
# FALLBACK AI ANALYSIS
# =========================================================

def fallback_insight(
    df,
    capacity,
    speed,
    results
):

    best_distance = min(
        results,
        key=lambda x:
        x["Distance (km)"]
    )


    best_time = min(
        results,
        key=lambda x:
        x["Delivery Time (min)"]
    )


    return f"""
### Pemahaman Masalah

Sistem melayani **{len(df)-1} customer** dengan total demand
**{df["demand"].iloc[1:].sum():.1f} kg**.

Kapasitas kendaraan adalah **{capacity:.1f} kg** dengan
kecepatan rata-rata **{speed:.1f} km/jam**.


### Perbandingan Skenario

Skenario dengan jarak perjalanan terendah adalah:

**{best_distance["Scenario"]}**

dengan total jarak:

**{best_distance["Distance (km)"]:.2f} km**.


Skenario dengan waktu pengiriman terendah adalah:

**{best_time["Scenario"]}**

dengan waktu:

**{best_time["Delivery Time (min)"]:.1f} menit**.


### Trade-off

Hasil simulasi menunjukkan bahwa strategi dengan jarak
lebih pendek tidak selalu menghasilkan waktu pengiriman
paling rendah karena waktu pelayanan customer juga
diperhitungkan.


### Decision Support

Hasil di atas dapat digunakan sebagai dasar untuk memilih
strategi berdasarkan prioritas perusahaan, seperti
meminimalkan jarak atau mempercepat pengiriman.

> Mode ini menggunakan analisis otomatis. AI eksternal
> dapat diaktifkan dengan memasukkan OpenAI API key.
"""


# =========================================================
# OPENAI ANALYSIS
# =========================================================

def ai_explain(
    df,
    capacity,
    speed,
    results,
    api_key
):

    if (
        not api_key
        or not OPENAI_AVAILABLE
    ):

        return fallback_insight(
            df,
            capacity,
            speed,
            results
        )


    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

    result_text = (
        pd.DataFrame(results)
        .drop(
            columns=[
                "_route_results"
            ],
            errors="ignore"
        )
        .round(2)
        .to_string(
            index=False
        )
    )


    prompt = f"""

Anda adalah AI decision-support assistant
untuk sistem optimasi rute pengiriman.

Data sistem:

Jumlah customer:
{len(df)-1}

Total demand:
{df["demand"].iloc[1:].sum():.2f} kg

Kapasitas kendaraan:
{capacity:.2f} kg

Kecepatan kendaraan:
{speed:.2f} km/jam


HASIL SIMULASI:

{result_text}


Buat analisis dalam Bahasa Indonesia
dengan empat bagian:

1. Pemahaman Masalah
2. Perbandingan Skenario
3. Trade-off
4. Rekomendasi Strategi

Gunakan hanya angka yang tersedia.

Jangan mengarang data.

Rekomendasi harus berdasarkan hasil
simulasi dan jelaskan alasannya.
"""


    response = client.responses.create(

        model="openrouter/free",

        input=prompt

    )


    return response.output_text


# =========================================================
# USER INTERFACE
# =========================================================

st.title("🚚 RouteAI")

st.caption(
    "AI-Powered Delivery Route Decision Support System"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ System Input"
    )

# ==========================
# DEPOT LOCATION
# ==========================

st.subheader("📍 Depot Location")

location_query = st.text_input(
    "Cari lokasi depot",
    placeholder="Contoh: Universitas Sumatera Utara, Medan"
)

if "depot_location" not in st.session_state:
    st.session_state.depot_location = None

if st.button("🔎 Cari Lokasi", key="search_location"):

    if location_query.strip():

        with st.spinner("Mencari lokasi..."):

            location = search_location(location_query)

            if location:
                st.session_state.depot_location = location
                st.success("Lokasi ditemukan!")
            else:
                st.error("Lokasi tidak ditemukan.")

if st.session_state.depot_location:
    depot_lat = st.session_state.depot_location["latitude"]
    depot_lon = st.session_state.depot_location["longitude"]

    st.caption(
        f"📍 {st.session_state.depot_location['display_name']}"
    )
else:
    depot_lat = 3.5952
    depot_lon = 98.6722
    
    capacity = st.number_input(

        "Vehicle Capacity (kg)",

        min_value=1.0,

        value=200.0,

        step=10.0

    )


    speed = st.number_input(

        "Average Speed (km/h)",

        min_value=1.0,

        value=40.0,

        step=5.0

    )


    st.subheader(
        "Depot"
    )


   depot_locations = {
    "Universitas Sumatera Utara (USU)": (3.5615, 98.6566),
    "Medan Mall": (3.5889, 98.6790),
    "Merdeka Walk": (3.5894, 98.6739),
    "Sun Plaza Medan": (3.5871, 98.6728),
    "Polonia": (3.5689, 98.6821)
}

selected_depot = st.selectbox(
    "📍 Pilih Lokasi Depot",
    list(depot_locations.keys())
)

depot_lat, depot_lon = depot_locations[
    selected_depot
]


    st.subheader(
        "🤖 AI"
    )


    api_key = st.sidebar.text_input(

        "OpenRouter API Key (optional)",

        type="password"

    )


# =========================================================
# CUSTOMER DATA
# =========================================================

st.subheader(
    "1. Customer Data"
)


default_data = pd.DataFrame({

    "customer":
        [
            "C1",
            "C2",
            "C3",
            "C4",
            "C5"
        ],

    "latitude":
        [
            3.6000,
            3.6060,
            3.5900,
            3.6120,
            3.5830
        ],

    "longitude":
        [
            98.6700,
            98.6800,
            98.6650,
            98.6900,
            98.6750
        ],

    "demand":
        [
            30,
            40,
            25,
            35,
            20
        ]

})


uploaded = st.file_uploader(

    "Upload CSV/Excel: customer, latitude, longitude, demand",

    type=[
        "csv",
        "xlsx"
    ]

)


if uploaded is not None:

    if uploaded.name.lower().endswith(
        ".csv"
    ):

        customer_df = pd.read_csv(
            uploaded
        )

    else:

        customer_df = pd.read_excel(
            uploaded
        )

else:

    customer_df = (
        default_data.copy()
    )


# =========================================================
# DATA VALIDATION
# =========================================================

customer_df.columns = [

    str(column)
    .strip()
    .lower()

    for column
    in customer_df.columns

]


required_columns = {

    "customer",
    "latitude",
    "longitude",
    "demand"

}


if not required_columns.issubset(
    customer_df.columns
):

    st.error(

        "Kolom wajib: "
        "customer, latitude, "
        "longitude, demand"

    )

    st.stop()


customer_df = customer_df[

    [
        "customer",
        "latitude",
        "longitude",
        "demand"
    ]

].copy()


customer_df["latitude"] = pd.to_numeric(

    customer_df["latitude"],

    errors="coerce"

)


customer_df["longitude"] = pd.to_numeric(

    customer_df["longitude"],

    errors="coerce"

)


customer_df["demand"] = pd.to_numeric(

    customer_df["demand"],

    errors="coerce"

)


customer_df = (

    customer_df
    .dropna()
    .reset_index(drop=True)

)


# =========================================================
# ADD DEPOT
# =========================================================

depot = pd.DataFrame([{

    "customer":
        "DEPOT",

    "latitude":
        depot_lat,

    "longitude":
        depot_lon,

    "demand":
        0.0

}])


df = pd.concat(

    [
        depot,
        customer_df
    ],

    ignore_index=True

)


st.dataframe(

    df,

    use_container_width=True

)


if len(df) < 2:

    st.warning(
        "Masukkan minimal 1 customer."
    )

    st.stop()


# =========================================================
# CAPACITY VALIDATION
# =========================================================

if (
    df["demand"]
    .iloc[1:]
    > capacity
).any():

    bad_customers = df.iloc[1:][

        df.iloc[1:]["demand"]
        > capacity

    ]


    st.error(

        "Demand customer berikut "
        "melebihi kapasitas kendaraan: "

        +

        ", ".join(
            bad_customers[
                "customer"
            ].astype(str)
        )

    )

    st.stop()


# =========================================================
# PROBLEM UNDERSTANDING
# =========================================================

st.subheader(
    "2. AI Problem Understanding"
)


st.info(

    f"""

Sistem mendeteksi **{len(df)-1} customer**.

Total demand:
**{df["demand"].iloc[1:].sum():.1f} kg**

Kapasitas kendaraan:
**{capacity:.1f} kg**

Kecepatan rata-rata:
**{speed:.1f} km/jam**

"""

)


# =========================================================
# RUN SIMULATION
# =========================================================

if st.button(

    "🚀 Generate & Run Scenarios",

    type="primary"

):

    with st.spinner(

        "Menghitung rute "
        "dan menjalankan simulasi..."

    ):

        scenarios, distance_matrix_data = (
            build_scenarios(
                df,
                capacity
            )
        )


        evaluations = []


        for (
            name,
            routes
        ) in scenarios.items():

            evaluations.append(

                evaluate_scenario(

                    name,

                    routes,

                    df,

                    distance_matrix_data,

                    speed,

                    capacity

                )

            )


        st.session_state[
            "df"
        ] = df


        st.session_state[
            "scenarios"
        ] = scenarios


        st.session_state[
            "distance_matrix"
        ] = distance_matrix_data


        st.session_state[
            "results"
        ] = evaluations


# =========================================================
# DISPLAY RESULTS
# =========================================================

if "results" in st.session_state:

    results = (
        st.session_state[
            "results"
        ]
    )


    scenarios = (
        st.session_state[
            "scenarios"
        ]
    )


    df = (
        st.session_state[
            "df"
        ]
    )


    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    st.subheader(
        "3. Scenario Comparison"
    )


    result_df = (

        pd.DataFrame(
            results
        )

        .drop(
            columns=[
                "_route_results"
            ],
            errors="ignore"
        )

    )


    st.dataframe(

        result_df.round(2),

        use_container_width=True

    )


    # -----------------------------------------------------
    # ROUTE MAP
    # -----------------------------------------------------

    st.subheader(
        "4. Route Visualization"
    )


    tabs = st.tabs(
        list(
            scenarios.keys()
        )
    )


    for (
        tab,
        (
            name,
            routes
        )
    ) in zip(
        tabs,
        scenarios.items()
    ):

        with tab:

            st.markdown(
                f"### {name}"
            )


            st_folium(

                make_map(
                    df,
                    routes
                ),

                width=None,

                height=500,

                key=f"map_{name}"

            )


            for (
                route_number,
                route
            ) in enumerate(
                routes,
                start=1
            ):

                labels = [

                    str(
                        df.loc[
                            i,
                            "customer"
                        ]
                    )

                    for i in route

                ]


                st.write(

                    f"Route {route_number}: "

                    +

                    " → ".join(
                        labels
                    )

                )


    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    st.subheader(
        "5. Performance Comparison"
    )


    chart_df = (

        result_df

        .set_index(
            "Scenario"
        )

        [

            [
                "Distance (km)",
                "Delivery Time (min)",
                "Throughput (cust/h)",
                "Utilization (%)"
            ]

        ]

    )


    st.bar_chart(
        chart_df
    )


    # -----------------------------------------------------
    # AI DECISION SUPPORT
    # -----------------------------------------------------

    st.subheader(
        "6. AI Decision Support"
    )


    if st.button(
        "🤖 Analyze Simulation Results"
    ):

        with st.spinner(
            "AI sedang menganalisis hasil simulasi..."
        ):

            try:

                insight = ai_explain(

                    df,

                    capacity,

                    speed,

                    results,

                    api_key

                )


                st.markdown(
                    insight
                )


            except Exception as error:

                st.error(
                    f"AI gagal dijalankan: {error}"
                )

                st.info(
                    "Pastikan API key benar "
                    "dan package OpenAI sudah "
                    "terpasang."
                )