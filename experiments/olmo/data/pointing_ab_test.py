from os.path import join

from olmo.data.dataset import Dataset, VIDEO_DATA_HOME

DATA = """
1	youtube-cc/youtube-cc-kw/qcmISGxah2A/qcmISGxah2A.mp4	98	150	women wearing red	cars
2	youtube-cc/youtube-cc-kw/TPBhWXKwvGU/TPBhWXKwvGU.mp4	0	56.066667	watch	water bottle
3	youtube-cc/youtube-cc-kw/cKihLVjyLOM/cKihLVjyLOM.mp4	0	57.5		
4	youtube-cc/youtube-cc-kw/24wD2mzbGz4/24wD2mzbGz4.mp4	0	58.3		
5	youtube-cc/youtube-cc-kw/9ssHnquqNcY/9ssHnquqNcY.mp4	23	83	pool	painting
6	youtube-cc/youtube-cc-kw/6l2w_aiHzz8/6l2w_aiHzz8.mp4	0	21.933333	hats	
7	youtube-cc/youtube-cc-temporal/GP6x0qpLB-U/GP6x0qpLB-U.mp4	33	93	cameras	
8	youtube-cc/youtube-cc-kw-2fps/AdCw8Si3gpc_2fps_0.000-13.500.mp4	0	13.5	blue wresler	
9	youtube-cc/youtube-cc-temporal/bZRY0B87F-0/bZRY0B87F-0.mp4	62	88		
10	youtube-cc/youtube-cc-kw/_2aZeHpwRgg/_2aZeHpwRgg.mp4	2	62	stairs	
11	youtube-cc/youtube-cc-kw-2fps/fbNpBqYDWVw_2fps_0.000-21.500.mp4	0	21.5	mushrroms	
12	video-caption-eval/vimeo/sports/vimeo_910058997.mp4	54	88	person	
13	youtube-cc/MammalNet/trimmed_video-2fps/y2vdr8mOj84_2fps_0.000-26.000.mp4	0	26	flipper	
14	youtube-cc/youtube-cc-kw/78fwvVm3TDE/78fwvVm3TDE.mp4	6	29	candy	
15	youtube-cc/youtube-cc-temporal/risrrovixmY/risrrovixmY.mp4	0	56.52		
16	youtube-cc/youtube-cc-temporal/suHQKvUobJw/suHQKvUobJw.mp4	26	67	women	
17	youtube-cc/youtube-cc-kw/3UQ0B9xGgQ8/3UQ0B9xGgQ8.mp4	0	59.9	buildings	
18	youtube-cc/youtube-cc-kw/w5ikeh72Wa0/w5ikeh72Wa0.mp4	0	39.933333	chairs	
19	youtube-cc/youtube-cc-temporal/U-6L2DV8WKU/U-6L2DV8WKU.mp4	29	87	street lights	
20	youtube-cc/youtube-cc-kw/44jNP6a8btM/44jNP6a8btM.mp4	0	18.083333	white text	
21	youtube-cc/youtube-cc-kw/aDMU4ZbcFU4/aDMU4ZbcFU4.mp4	12	35	crayfish	
22	youtube-cc/youtube-cc-temporal/HpSKSOQCv9U/HpSKSOQCv9U.mp4	24	84	signs	
23	youtube-cc/youtube-cc-kw/xqE_HXjjDi8/xqE_HXjjDi8.mp4	0	48.6486	buildings	
24	youtube-cc/youtube-cc-kw/KPFlNNHqbuE/KPFlNNHqbuE.mp4	52	93	pier	
25	youtube-cc/youtube-cc-kw/yGRJotaJ4wA/yGRJotaJ4wA.mp4	0	35.041667	people	
26	youtube-cc/youtube-cc-temporal/Ptuugt3iTn4/Ptuugt3iTn4.mp4	49	109		
27	youtube-cc/youtube-cc-kw/q2PlU_CTf14/q2PlU_CTf14.mp4	26	85	hulk	
28	youtube-cc/youtube-cc-kw/tO9HG-GF5sM/tO9HG-GF5sM.mp4	0	40	"shooting a basket ball"	
29	youtube-cc/youtube-cc-temporal/LopadPyAJ4c/LopadPyAJ4c.mp4	2	42		
30	youtube-cc/youtube-cc-kw/mUkG9MaBcZw/mUkG9MaBcZw.mp4	0	47.266667	planet	
31	video-caption-eval/vimeo/documentary/vimeo_1053742537.mp4	17	77	airplanes	
32	youtube-cc/youtube-cc-kw/R2DUTmF1gcM/R2DUTmF1gcM.mp4	64	109	stars	
33	youtube-cc/youtube-cc-temporal/ejXYepaD_QU/ejXYepaD_QU.mp4	6	38	person wearing glasses	
34	youtube-cc/youtube-cc-kw/9kr8NTbblPE/9kr8NTbblPE.mp4	0	34.766667	phones	
35	youtube-cc/youtube-cc-kw/MyR3JZOxCis/MyR3JZOxCis.mp4	0	23.8		
36	youtube-cc/youtube-cc-kw/LuFhLkEdOlg/LuFhLkEdOlg.mp4	0	23.033333	train stop	
37	youtube-cc/youtube-cc-kw/zEkfmXIsSCA/zEkfmXIsSCA.mp4	0	49.433333		
38	youtube-cc/youtube-cc-temporal/MMNpSVKM44k/MMNpSVKM44k.mp4	0	43.063333	penguins	
39	youtube-cc/youtube-cc-temporal/DiK4BNxAI-s/DiK4BNxAI-s.mp4	14	54		
40	youtube-cc/youtube-cc-kw/C9z0KZwFny8/C9z0KZwFny8.mp4	2	39	podiums	
41	youtube-cc/youtube-cc-temporal/ZDHUX8Mjw0c/ZDHUX8Mjw0c.mp4	0	23.556867	cars	
42	youtube-cc/youtube-cc-exist-2fps/2vm2A6W8b-s_2fps_20.000-35.012.mp4	0	15.5	metal cups	
43	youtube-cc/youtube-cc-kw/rq2hcq5aC3g/rq2hcq5aC3g.mp4	0	39.333333	crosses	
44	youtube-cc/youtube-cc-kw/fXLIYtHziBQ/fXLIYtHziBQ.mp4	0	40.1	windows	
45	youtube-cc/youtube-cc-temporal/zXDJEOUmKyI/zXDJEOUmKyI.mp4	43	99	jars	
46	video-caption-eval/vimeo/nature-animal/vimeo_1021151113.mp4	0	45.9	moose	animals
47	youtube-cc/youtube-cc-kw/XBsrHJ1bhko/XBsrHJ1bhko.mp4	49	109		
48	video-caption-eval/vimeo/performance/vimeo_1081540548.mp4	47	84	dancer	
49	youtube-cc/youtube-cc-kw/u_dwObj2zg0/u_dwObj2zg0.mp4	0	22.966667		
50	youtube-cc/youtube-cc-kw/YJK5dotNLOU/YJK5dotNLOU.mp4	0	23.933333	candles	hands
51	youtube-cc/youtube-cc-kw/xUmOvxWkwlk/xUmOvxWkwlk.mp4	0	36.633333	belts	
52	youtube-cc/youtube-cc-kw/S6rYnn3KXx8/S6rYnn3KXx8.mp4	9	69	cups	
53	youtube-cc/youtube-cc-exist-2fps/wxZQMgMMTLg_2fps_42.000-69.009.mp4	0	27.5	robots	
54	youtube-cc/youtube-cc-kw/aZZXkbcTtBw/aZZXkbcTtBw.mp4	0	17.5		
55	youtube-cc/youtube-cc-temporal-2fps/IMT0gUx6QxQ_2fps_45.000-69.782.mp4	0	25		
56	video-caption-eval/vimeo/nature-animal/vimeo_229157180.mp4	0	45.336958	whales	scuba diver
57	youtube-cc/youtube-cc-kw/VrE9TVnDQp8/VrE9TVnDQp8.mp4	0	15.1151	dogs	
58	youtube-cc/youtube-cc-kw/o6Bgb5uD3w8/o6Bgb5uD3w8.mp4	0	52	helmets	baseball players
59	youtube-cc/youtube-cc-temporal/KMQFrWVmMqQ/KMQFrWVmMqQ.mp4	22	75	road	
60	youtube-cc/youtube-cc-kw/hz_2646i47c/hz_2646i47c.mp4	0	26.36		
61	youtube-cc/youtube-cc-kw/WjajlEjCRec/WjajlEjCRec.mp4	0	30.397033		
62	youtube-cc/youtube-cc-kw/u9ty3NJT60o/u9ty3NJT60o.mp4	65	87	pipes	
63	video-caption-eval/ego4d/6d0542b2-34c1-4ad9-96f9-7df82e246045.mp4	42	63	wheels	
64	youtube-cc/youtube-cc-temporal/HdysXPZ_Ks8/HdysXPZ_Ks8.mp4	65	115	pizzas	
65	youtube-cc/MammalNet/trimmed_video-2fps/2bNkFFsRndU_2fps_0.000-15.000.mp4	0	15		
66	youtube-cc/youtube-cc-kw/lyEiV-Ilc2E/lyEiV-Ilc2E.mp4	0	28.8		
67	youtube-cc/youtube-cc-temporal/5OUip3Hs7rA/5OUip3Hs7rA.mp4	27	64	person playing an instrument	
68	youtube-cc/youtube-cc-temporal-2fps/ok6l4c3_h-U_2fps_0.000-53.500.mp4	0	53.5		
69	youtube-cc/youtube-cc-kw/zL3eCzFk-iQ/zL3eCzFk-iQ.mp4	13	41	house	
70	youtube-cc/youtube-cc-kw/d-KMFHXKCdk/d-KMFHXKCdk.mp4	0	16.9		
71	youtube-cc/youtube-cc-temporal/2UCRVBO_fNA/2UCRVBO_fNA.mp4	0	60	ghosts	
72	youtube-cc/youtube-cc-kw/7r148IVyCo4/7r148IVyCo4.mp4	25	85	construction workers	
73	youtube-cc/youtube-cc-kw/kqMsTLXurfs/kqMsTLXurfs.mp4	69	107	trains	
74	youtube-cc/youtube-cc-kw/SPGy0EcySEE/SPGy0EcySEE.mp4	0	11.966667	trees	
75	video-caption-eval/vimeo/travel/vimeo_1055159864.mp4	0	15.1		
76	youtube-cc/youtube-cc-temporal/GugjmFEruFg/GugjmFEruFg.mp4	41	72	bowls	
77	youtube-cc/youtube-cc-temporal/8zEjR64fLKI/8zEjR64fLKI.mp4	0	24.1238		
78	youtube-cc/youtube-cc-kw/R16B4MuJO84/R16B4MuJO84.mp4	0	14.433333		
79	youtube-cc/youtube-cc-kw-2fps/x5xDRv1IDtU_2fps_0.000-20.500.mp4	0	20.5	red players	
80	youtube-cc/youtube-cc-kw/6-OACwBunjY/6-OACwBunjY.mp4	14	38		
81	youtube-cc/youtube-cc-kw/bna_ZFSRJ48/bna_ZFSRJ48.mp4	0	55.72		
82	youtube-cc/youtube-cc-kw/TVCMR6hdL2k/TVCMR6hdL2k.mp4	0	37.733333	video game characters	word crouch
83	youtube-cc/youtube-cc-kw/-mTtIIIMIr4/-mTtIIIMIr4.mp4	0	31.566667		
84	youtube-cc/youtube-cc-kw/NVw1VU72Afs/NVw1VU72Afs.mp4	0	30.333333		
85	youtube-cc/youtube-cc-temporal/1iUbZvCQLEs/1iUbZvCQLEs.mp4	138	165		
86	youtube-cc/youtube-cc-temporal/7oVFJX0QVCU/7oVFJX0QVCU.mp4	113	150	people dancing	
87	youtube-cc/youtube-cc-temporal/PUW2ek0GuM0/PUW2ek0GuM0.mp4	9	69	cars	
88	youtube-cc/youtube-cc-kw/WoL8ZEb0YYI/WoL8ZEb0YYI.mp4	0	56.933333	lighting bolts	
89	youtube-cc/youtube-cc-exist/I1Bdp2tMFsY/I1Bdp2tMFsY.mp4	18	78	jewlery	
90	video-caption-eval/vimeo/beauty-fashion/vimeo_456995518.mp4	0	20		
91	youtube-cc/youtube-cc-temporal/bFyEr8vk2SY/bFyEr8vk2SY.mp4	2	62		
92	youtube-cc/youtube-cc-kw/p98l3sYYnCs/p98l3sYYnCs.mp4	46	86		
93	youtube-cc/youtube-cc-temporal/05BhWJQ9tuo/05BhWJQ9tuo.mp4	13	73	scarves	
94	youtube-cc/youtube-cc-temporal-2fps/Tn3jzg6nhlU_2fps_0.000-58.500.mp4	0	58.5	fire hoses	bridges
95	youtube-cc/youtube-cc-kw/MjWG6aNXyi8/MjWG6aNXyi8.mp4	0	19.334411	take picture button	
96	youtube-cc/youtube-cc-kw/Q8k_vLCCy10/Q8k_vLCCy10.mp4	2	62	weights	
97	youtube-cc/MammalNet/trimmed_video-2fps/400dUnIsOsE_2fps_0.000-53.000.mp4	0	53	waves	birds
98	youtube-cc/youtube-cc-temporal/FjzIG7hV9ac/FjzIG7hV9ac.mp4	8	33		
99	youtube-cc/youtube-cc-temporal/on5uO3Fc8JQ/on5uO3Fc8JQ.mp4	21	81	doors	
100	youtube-cc/youtube-cc-temporal/7u83nFcvtrM/7u83nFcvtrM.mp4	55	115	boats	
101	youtube-cc/youtube-cc-kw-2fps/1bDQC8sYyGc_2fps_0.000-36.000.mp4	0	36	women wearing pink	
102	youtube-cc/youtube-cc-kw/FqHe8QwbGf0/FqHe8QwbGf0.mp4	0	12.833333	gaint roses	
103	youtube-cc/youtube-cc-kw/bolUy5E8WUc/bolUy5E8WUc.mp4	0	38.666667		
104	youtube-cc/youtube-cc-kw/dXdHYHkH3ow/dXdHYHkH3ow.mp4	6	56	numbers	planets
105	youtube-cc/youtube-cc-kw/ey2GIzkAyZw/ey2GIzkAyZw.mp4	0	34.833333		
106	youtube-cc/youtube-cc-kw-2fps/4w-GX5E4sV8_2fps_0.000-26.000.mp4	0	26	shells	
107	youtube-cc/youtube-cc-kw/9Xdnbz6I4NI/9Xdnbz6I4NI.mp4	0	21.388033		
108	youtube-cc/youtube-cc-temporal/h7UOUCV5qZA/h7UOUCV5qZA.mp4	4	30	clouds	
109	youtube-cc/youtube-cc-kw/buIJ3W-vH4Q/buIJ3W-vH4Q.mp4	0	54.966667		
110	video-caption-eval/vimeo/virtual-tour/vimeo_998263081.mp4	47	99	plates	wine
111	youtube-cc/youtube-cc-kw/RVq8JyifCHE/RVq8JyifCHE.mp4	18	40		
112	youtube-cc/youtube-cc-kw/wCydQtVZz-Y/wCydQtVZz-Y.mp4	7	67	cartoon person	
113	youtube-cc/youtube-cc-temporal/QN8SSjBM-Tc/QN8SSjBM-Tc.mp4	18	78	percent sign	
114	youtube-cc/youtube-cc-kw/wFRpRLLpayY/wFRpRLLpayY.mp4	0	17.666667		
115	youtube-cc/youtube-cc-kw/yMo0ezyy48A/yMo0ezyy48A.mp4	17	77	tower	
116	video-caption-eval/ego4d/grp-f30e01bb-e8f1-42d2-922c-aa7b66da9035_bounded_decimal_2_204.43_321.91.mp4	43	95	container	
117	youtube-cc/youtube-cc-temporal/gPRRJz5dL0E/gPRRJz5dL0E.mp4	5	54	doughnut	
118	youtube-cc/youtube-cc-kw/dDLIboNMB7s/dDLIboNMB7s.mp4	42	84	music stand	
119	youtube-cc/youtube-cc-kw/l_sKP7xxX8M/l_sKP7xxX8M.mp4	0	16.866667	bowl	
120	youtube-cc/youtube-cc-kw/0pkIIlIAFYY/0pkIIlIAFYY.mp4	8	52		
121	youtube-cc/youtube-cc-kw/uauxLIBnT88/uauxLIBnT88.mp4	16	60		
122	youtube-cc/youtube-cc-kw/hhgqP9OcC5w/hhgqP9OcC5w.mp4	0	41.958333	chairs	scientists
123	youtube-cc/youtube-cc-kw/2dPn0I5jNXw/2dPn0I5jNXw.mp4	0	60	book	
124	youtube-cc/youtube-cc-temporal/aVBZVxBeQt0/aVBZVxBeQt0.mp4	50	109	necklace	
125	youtube-cc/youtube-cc-kw/2qogUiAnPb0/2qogUiAnPb0.mp4	0	47	statue	tree
126	video-caption-eval/vimeo/daily-people/vimeo_115964481.mp4	22	71	boxes	backpacks
127	youtube-cc/youtube-cc-kw/spMzVQWqUJE/spMzVQWqUJE.mp4	0	49.958333	axe	
128	youtube-cc/youtube-cc-kw/9IfY5Rg4UVA/9IfY5Rg4UVA.mp4	19	56	mountain	
129	youtube-cc/youtube-cc-kw/twmo8froAwI/twmo8froAwI.mp4	79	110		
130	youtube-cc/youtube-cc-kw/iTH08NJw47Y/iTH08NJw47Y.mp4	0	19.72		
131	youtube-cc/youtube-cc-kw/buBktkggoY8/buBktkggoY8.mp4	0	19.561208	people	
132	youtube-cc/youtube-cc-kw/xH55lxyOK2k/xH55lxyOK2k.mp4	0	32.533333	man with red tie	
133	youtube-cc/youtube-cc-kw/_2cAuxJPW1g/_2cAuxJPW1g.mp4	5	65		
134	youtube-cc/youtube-cc-kw/8cG9yWdIHUc/8cG9yWdIHUc.mp4	31	91		
135	youtube-cc/youtube-cc-kw/96ukV2TQORw/96ukV2TQORw.mp4	2	62		
136	youtube-cc/youtube-cc-kw/VE25kJeGn1U/VE25kJeGn1U.mp4	0	23	USA	
137	youtube-cc/youtube-cc-kw/Upozkv5kCyU/Upozkv5kCyU.mp4	0	51.7	stethoscope	
138	youtube-cc/youtube-cc-kw/ExotlMPVXHs/ExotlMPVXHs.mp4	0	44.4		
139	youtube-cc/youtube-cc-kw/LdXXA9KCRAk/LdXXA9KCRAk.mp4	54	95	bike	
140	video-caption-eval/vimeo/handicraft-factory/vimeo_243960266.mp4	15	41	car	
141	youtube-cc/youtube-cc-kw/iDcWbzbyNFA/iDcWbzbyNFA.mp4	25	85		
142	youtube-cc/youtube-cc-temporal-2fps/E6XV5J1Qxic_2fps_10.000-29.455.mp4	0	19.5	ambulance	traffic cone
143	youtube-cc/youtube-cc-kw/gJ5rP2S_XG0/gJ5rP2S_XG0.mp4	0	46.833333	door	
144	youtube-cc/youtube-cc-temporal-2fps/DPuQWUwRBrg_2fps_0.000-62.025.mp4	28	54		
145	video-caption-eval/bdd100k/8f86a3ed-8c368fc2.mov	0	20.315		
146	youtube-cc/youtube-cc-temporal-2fps/BL83CZzvS28_2fps_3.500-19.213.mp4	0	16		
147	youtube-cc/youtube-cc-kw/XPCwMV_UMdc/XPCwMV_UMdc.mp4	0	59.083333	parrot	
148	youtube-cc/youtube-cc-kw/wrH9DFXBtCY/wrH9DFXBtCY.mp4	4	62	building	
149	youtube-cc/youtube-cc-kw/pjs-5MY7SjU/pjs-5MY7SjU.mp4	0	18.933333	king	
150	youtube-cc/youtube-cc-kw/x_Zy2mymAdg/x_Zy2mymAdg.mp4	0	46.5		
151	youtube-cc/youtube-cc-kw/QnnWqLrOe2c/QnnWqLrOe2c.mp4	117	148		
152	youtube-cc/youtube-cc-exist-2fps/3llBcLC3S4k_2fps_40.500-64.399.mp4	0	24		
153	youtube-cc/youtube-cc-kw/ZgVSWAuLlrc/ZgVSWAuLlrc.mp4	0	18.966667	white hat	
154	youtube-cc/youtube-cc-temporal/gFginAiKD0E/gFginAiKD0E.mp4	2	57		
155	video-caption-eval/bdd100k/aee39fd5-9ba7f525.mov	0	40.238333		
156	video-caption-eval/ego4d/bb16e081-d56d-40c0-a000-6d59ddb4498d_bounded_decimal_2_16.27_70.94.mp4	0	54.666667		
157	youtube-cc/youtube-cc-kw/KqYNy-FWtts/KqYNy-FWtts.mp4	0	52.84	blue squares	shoes
158	youtube-cc/youtube-cc-temporal/uFghNawGJRc/uFghNawGJRc.mp4	17	77		
159	youtube-cc/youtube-cc-kw/bv4SoQJyCNI/bv4SoQJyCNI.mp4	0	25.033333	tennis court	
160	youtube-cc/youtube-cc-kw/vVRqG6Eyd40/vVRqG6Eyd40.mp4	38	98		
161	youtube-cc/youtube-cc-temporal/t0lcm62xT4w/t0lcm62xT4w.mp4	73	118	super hero	
162	youtube-cc/youtube-cc-kw/8awBbSfhmzM/8awBbSfhmzM.mp4	0	51.133333	crown	
163	youtube-cc/youtube-cc-temporal/QvJUVcTPyB8/QvJUVcTPyB8.mp4	54	114		
164	youtube-cc/youtube-cc-temporal/REeKF4-uwGo/REeKF4-uwGo.mp4	36	96		
165	youtube-cc/youtube-cc-kw/Mty39hQhWJk/Mty39hQhWJk.mp4	0	18		
166	youtube-cc/youtube-cc-kw/f1WJELHzyMQ/f1WJELHzyMQ.mp4	0	38.4		
167	youtube-cc/youtube-cc-kw/UQETY5ghZKk/UQETY5ghZKk.mp4	0	38.136333		
168	youtube-cc/youtube-cc-kw/_e0okdxAq0E/_e0okdxAq0E.mp4	37	60		
169	youtube-cc/youtube-cc-temporal/S20caR4A5_o/S20caR4A5_o.mp4	18	58		
170	youtube-cc/youtube-cc-kw-2fps/OcycNsNhD9w_2fps_0.000-46.000.mp4	0	46		
171	youtube-cc/youtube-cc-temporal/5XG4QCi3pw4/5XG4QCi3pw4.mp4	99	159		
172	video-caption-eval/vimeo/object/vimeo_107702153.mp4	123	170		
173	youtube-cc/youtube-cc-kw/0Yy0nDc8ZBQ/0Yy0nDc8ZBQ.mp4	8	68		
174	youtube-cc/youtube-cc-kw/V89_vTTRKnI/V89_vTTRKnI.mp4	6	63		
175	youtube-cc/youtube-cc-temporal/yBMh5iNFFTA/yBMh5iNFFTA.mp4	16	66		
176	youtube-cc/youtube-cc-temporal/S74e6RuD7Vo/S74e6RuD7Vo.mp4	64	114		
177	youtube-cc/youtube-cc-kw-2fps/HAj8dE-UiKo_2fps_0.000-43.500.mp4	0	43.5		
178	video-caption-eval/vimeo/documentary/vimeo_800931196.mp4	0	58.066667		
179	youtube-cc/youtube-cc-temporal-2fps/TfV8s-MhiYY_2fps_20.000-52.785.mp4	0	33		
180	video-caption-eval/vimeo/game/vimeo_1083797954.mp4	0	37.966667		
181	youtube-cc/youtube-cc-temporal-2fps/heBQHri8ZPk_2fps_40.500-67.509.mp4	0	27.5		
182	video-caption-eval/vimeo/travel/vimeo_1015699838.mp4	0	56.222833		
183	youtube-cc/youtube-cc-kw/KH31RzovrXU/KH31RzovrXU.mp4	0	40.566667		
184	youtube-cc/youtube-cc-kw/c2HXQ42MrOA/c2HXQ42MrOA.mp4	9	55		
185	youtube-cc/youtube-cc-kw/ndV_pkV4F4g/ndV_pkV4F4g.mp4	0	17.4		
186	youtube-cc/youtube-cc-kw/T7L2pm0g1CU/T7L2pm0g1CU.mp4	16	76		
187	youtube-cc/youtube-cc-kw/TxCLU1E-OKk/TxCLU1E-OKk.mp4	0	34.033333		
188	youtube-cc/youtube-cc-kw/RZhMgim0ucc/RZhMgim0ucc.mp4	11	57		
189	video-caption-eval/vimeo/object/vimeo_933799307.mp4	0	59.592867		
190	video-caption-eval/vimeo/documentary/vimeo_129102165.mp4	0	26.48		
191	youtube-cc/youtube-cc-temporal/ZIpv0LtZ4BE/ZIpv0LtZ4BE.mp4	29	67		
192	video-caption-eval/vimeo/CG-Animation/vimeo_1063850055.mp4	10	49		
193	youtube-cc/youtube-cc-temporal-2fps/NqLpEQCNwCc_2fps_0.000-43.500.mp4	0	43.5		
194	youtube-cc/youtube-cc-kw/M_OZc4M06mA/M_OZc4M06mA.mp4	0	16.765467		
195	youtube-cc/youtube-cc-kw/Mx82jTOT5CY/Mx82jTOT5CY.mp4	0	11.1		
196	youtube-cc/youtube-cc-kw/9aEEvzz-iXw/9aEEvzz-iXw.mp4	18	38		
197	youtube-cc/youtube-cc-kw/KdMQZGZceYQ/KdMQZGZceYQ.mp4	66	86		
198	youtube-cc/youtube-cc-kw/TQXjOHpG_7s/TQXjOHpG_7s.mp4	49	77		
199	youtube-cc/youtube-cc-kw/m8jv3lcgHqU/m8jv3lcgHqU.mp4	1	58		
"""


class ABPointTestData(Dataset):
    def __init__(self):
        data = []
        for row in DATA.strip().split("\n"):
            parts = row.split("\t")
            video_id, video, start, end = parts[:4]
            queries = parts[4:]
            video_path = join(VIDEO_DATA_HOME, video)
            for q in queries:
                if q:
                    data.append(dict(
                        video=video_path,
                        question=f"Point to the {q}.",
                        metadata=dict(video_path=video_path, id=video_id, query=q,
                                      clip_start_time=float(start), clip_end_time=float(end)),
                        style="user_qa"
                    ))
        self.data = data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


if __name__ == '__main__':
    print(len(ABPointTestData()))