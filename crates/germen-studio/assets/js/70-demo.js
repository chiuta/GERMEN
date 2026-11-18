// 70-demo.js — the demonstration page. The interface is the real one; what answers it is this file: a
// training of the smallest model, recorded once on a real computer (scripts/record_demo.py) and played
// back at the pace it happened, the text that model wrote, and two speed measurements. Around them are
// plain stand-ins: a few folders with a few texts, a list of models that grows when a "training" ends.
// Nothing is computed and nothing leaves the page. Only the demonstration build contains this file.

/* RECORDING:BEGIN (written by scripts/record_demo.py from a real run; do not edit by hand) */
const REC = {
  system: {"data_dir":"/home/demo/GERMEN/data","languages":["en","ro","fr","it","es","pt","de","hu"],"limits":{"max_threads":1024,"prompt_bytes":65536,"upload_bytes":2147483648},"logical_cpus":2,"memory":{"available":17179869184,"limit":15461882265},"models_dir":"/home/demo/GERMEN/models","optimised":true,"platform":"linux/x86_64","presets":[{"d_model":128,"heads":4,"layers":4,"name":"nano","params":853120,"train_bytes":537937920,"weights_bytes":3412480},{"d_model":192,"heads":6,"layers":6,"name":"micro","params":2755008,"train_bytes":1124113408,"weights_bytes":11020032},{"d_model":384,"heads":6,"layers":6,"name":"mini","params":10818432,"train_bytes":1800484864,"weights_bytes":43273728},{"d_model":512,"heads":8,"layers":8,"name":"small","params":25436672,"train_bytes":3238141952,"weights_bytes":101746688},{"d_model":768,"heads":12,"layers":12,"name":"base","params":85347072,"train_bytes":7636037632,"weights_bytes":341388288}],"simd":"AVX-512+AVX2+FMA","version":"0.5.0","workers":2,"workspace":"/home/demo/GERMEN"},
  model: {"best_val_loss":1.45761,"bytes":11219897,"context":256,"corpus_bytes":524302,"d_model":128,"elapsed_secs":46.2519,"has_manifest":true,"heads":4,"last_train_loss":1.44275,"last_val_loss":1.45761,"layers":4,"name":"demo-nano","params":933760,"path":"/home/demo/GERMEN/models/demo-nano","step":240,"tokenizer":"bpe","tokens_seen":491520,"vocab_size":571},
  train: {"request":{"accum":1,"batch":8,"bpe_sample_mb":64,"data":{"kind":"demo","mb":0.5,"seed":1},"eval_batches":8,"eval_every":40,"grad_clip":1,"log_every":5,"lr":0.0006,"min_lr_ratio":0.1,"mode":"new","name":"demo-nano","preset":"nano","replicas":0,"sample_every":60,"save_every":500,"seed":1,"steps":240,"tokenizer":"bpe","val_fraction":0.05,"vocab":2048,"warmup":12,"weight_decay":0.01},"events":[[0.0,{"done":null,"name":"reading","total":null,"type":"stage"}],[0.003,{"done":0,"name":"bpe","total":1792,"type":"stage"}],[0.006,{"done":16,"name":"bpe","total":1792,"type":"stage"}],[0.006,{"done":null,"name":"corpus","total":null,"type":"stage"}],[0.027,{"bytes_per_token":5.94285,"context":256,"corpus_bytes":524302,"corpus_tokens":88224,"d_model":128,"est_memory_bytes":329585152,"from_step":0,"heads":4,"layers":4,"optimizer_restored":false,"params":933760,"replicas":2,"resume":false,"tokenizer":"bpe","tokens_per_step":2048,"total_steps":240,"train_tokens":83786,"type":"start","vocab_size":571,"workers":2}],[1.143,{"elapsed":1.11838,"eta":52.5639,"grad_norm":3.44304,"loss":6.15932,"lr":0.00025,"step":5,"total":240,"tps":9156.1,"type":"step"}],[2.121,{"elapsed":2.08945,"eta":48.0574,"grad_norm":2.62709,"loss":5.59092,"lr":0.0005,"step":10,"total":240,"tps":10545.1,"type":"step"}],[3.107,{"elapsed":3.07896,"eta":46.1844,"grad_norm":2.08173,"loss":4.91367,"lr":0.000599897,"step":15,"total":240,"tps":10348.6,"type":"step"}],[4.078,{"elapsed":4.05194,"eta":44.5713,"grad_norm":1.82748,"loss":4.39291,"lr":0.000598745,"step":20,"total":240,"tps":10524.4,"type":"step"}],[5.025,{"elapsed":4.99942,"eta":42.995,"grad_norm":1.70427,"loss":3.96493,"lr":0.000596318,"step":25,"total":240,"tps":10807.5,"type":"step"}],[5.937,{"elapsed":5.91147,"eta":41.3803,"grad_norm":1.57548,"loss":3.58265,"lr":0.000592626,"step":30,"total":240,"tps":11227.5,"type":"step"}],[6.877,{"elapsed":6.8511,"eta":40.1278,"grad_norm":1.45975,"loss":3.22847,"lr":0.000587689,"step":35,"total":240,"tps":10898.0,"type":"step"}],[7.777,{"elapsed":7.75074,"eta":38.7537,"grad_norm":1.31882,"loss":2.90625,"lr":0.00058153,"step":40,"total":240,"tps":11382.3,"type":"step"}],[8.393,{"best":true,"bpb":0.662471,"step":40,"type":"eval","val_loss":2.7289}],[9.368,{"elapsed":8.72355,"eta":37.802,"grad_norm":1.1878,"loss":2.62782,"lr":0.000574176,"step":45,"total":240,"tps":10526.2,"type":"step"}],[10.28,{"elapsed":9.63822,"eta":36.6252,"grad_norm":1.02538,"loss":2.38963,"lr":0.000565665,"step":50,"total":240,"tps":11195.3,"type":"step"}],[11.141,{"elapsed":10.499,"eta":35.3147,"grad_norm":0.874092,"loss":2.18897,"lr":0.000556035,"step":55,"total":240,"tps":11896.6,"type":"step"}],[12.057,{"elapsed":11.4144,"eta":34.2431,"grad_norm":0.763242,"loss":2.01803,"lr":0.000545333,"step":60,"total":240,"tps":11186.5,"type":"step"}],[12.116,{"step":60,"text":". Copiii repară o poveste lungă. Meșterii așteaptă un pod de piatră. Păsările ascultă un copac înalt, dar caută o poveste lungă. \nCopiii ascultă o poveste lungă pe acasă. \nAstronomul construiește o carte veche. Croitoreasa desenează un carte veche. \nAna caută un copac înalt, și repară luna plină pe munte. Cercetătorii repară un pod de piatră. Vulpea găsește luna plină. Mihai repară un cântec frumos, apoi poartă așteaptă harta orașului, iar apoi. \nCopilul o lampă albastră. \nProfesorul desenează un telescop mic seara. \n de piatră, dar citește o carte veche, apoi așteaptă un cântec frumos. Doctorița găsește dar construiește o carte veche acasă. Pescarul așteaptă o poveste lungă. Profesorul desenează un pod de piatră lângă fereastră, iar apoi desenează harta orașului, și găsește un ascultă vechea fântână. Croitoreasa poartă un telescop","type":"sample"}],[13.062,{"elapsed":12.3605,"eta":33.2782,"grad_norm":0.621872,"loss":1.90457,"lr":0.000533609,"step":65,"total":240,"tps":10823.2,"type":"step"}],[14.077,{"elapsed":13.3704,"eta":32.471,"grad_norm":0.539659,"loss":1.80822,"lr":0.000520919,"step":70,"total":240,"tps":10139.1,"type":"step"}],[15.002,{"elapsed":14.2999,"eta":31.4597,"grad_norm":0.454934,"loss":1.73474,"lr":0.000507323,"step":75,"total":240,"tps":11017.2,"type":"step"}],[15.853,{"elapsed":15.1514,"eta":30.3029,"grad_norm":0.407324,"loss":1.68333,"lr":0.000492886,"step":80,"total":240,"tps":12025.2,"type":"step"}],[16.486,{"best":true,"bpb":0.403212,"step":80,"type":"eval","val_loss":1.66094}],[17.42,{"elapsed":16.0856,"eta":29.3326,"grad_norm":0.35971,"loss":1.64705,"lr":0.000477676,"step":85,"total":240,"tps":10961.1,"type":"step"}],[18.36,{"elapsed":17.0256,"eta":28.3761,"grad_norm":0.334603,"loss":1.62654,"lr":0.000461765,"step":90,"total":240,"tps":10893.7,"type":"step"}],[19.314,{"elapsed":17.979,"eta":27.4417,"grad_norm":0.311243,"loss":1.6016,"lr":0.00044523,"step":95,"total":240,"tps":10740.5,"type":"step"}],[20.289,{"elapsed":18.9504,"eta":26.5306,"grad_norm":0.326098,"loss":1.57734,"lr":0.000428147,"step":100,"total":240,"tps":10541.5,"type":"step"}],[21.279,{"elapsed":19.9444,"eta":25.6428,"grad_norm":0.310377,"loss":1.56993,"lr":0.000410599,"step":105,"total":240,"tps":10302.0,"type":"step"}],[22.168,{"elapsed":20.8333,"eta":24.6211,"grad_norm":0.316776,"loss":1.54937,"lr":0.000392668,"step":110,"total":240,"tps":11520.3,"type":"step"}],[23.14,{"elapsed":21.8046,"eta":23.7007,"grad_norm":0.290664,"loss":1.53843,"lr":0.000374441,"step":115,"total":240,"tps":10542.1,"type":"step"}],[24.117,{"elapsed":22.7799,"eta":22.7799,"grad_norm":0.328588,"loss":1.52092,"lr":0.000356002,"step":120,"total":240,"tps":10499.5,"type":"step"}],[24.777,{"best":true,"bpb":0.367646,"step":120,"type":"eval","val_loss":1.51443}],[24.838,{"step":120,"text":". Meșterii construiesc un copac înalt, iar apoi repară pădurea întunecată. Inginerul găsește un copac înalt. \nCopilul construiește o scrisoare, dar poartă o lampă albastră. Pescarul repară vechea fântână. Ana găsește vechea fântână, și caută luna plină. Pisica citește luna plină pe malul râului. Păsările poartă o scrisoare, și ascultă un cântec frumos. \nDoctorița găsește o carte veche. Inginerul ascultă un cântec frumos. \nAstronomul ascultă un copac înalt. Cercetătorii văd o lampă albastră seara. Profesorul așteaptă pădurea întunecată. Ana construiește un pod de piatră, apoi poartă vechea fântână. Meșterii desenează o scrisoare lângă fereastră. Cercetătorii ascultă un telescop mic. \nMeșterii găsesc vechea fântână. Croitoreasa găsește un pod de piatră, dar vede un pod de piatră. Copiii văd un copac înalt. Copiii așteaptă o carte veche lângă fereastră. \nCroitoreasa repară pădurea","type":"sample"}],[25.881,{"elapsed":23.8182,"eta":21.9127,"grad_norm":0.292738,"loss":1.51757,"lr":0.00033744,"step":125,"total":240,"tps":9862.51,"type":"step"}],[26.85,{"elapsed":24.7918,"eta":20.9776,"grad_norm":0.279799,"loss":1.50043,"lr":0.000318842,"step":130,"total":240,"tps":10517.8,"type":"step"}],[27.735,{"elapsed":25.677,"eta":19.971,"grad_norm":0.278161,"loss":1.50244,"lr":0.000300298,"step":135,"total":240,"tps":11567.0,"type":"step"}],[28.675,{"elapsed":26.617,"eta":19.0122,"grad_norm":0.285479,"loss":1.49895,"lr":0.000281894,"step":140,"total":240,"tps":10893.9,"type":"step"}],[29.651,{"elapsed":27.5923,"eta":18.0777,"grad_norm":0.299636,"loss":1.48847,"lr":0.000263719,"step":145,"total":240,"tps":10500.0,"type":"step"}],[30.602,{"elapsed":28.5437,"eta":17.1262,"grad_norm":0.266958,"loss":1.50373,"lr":0.000245858,"step":150,"total":240,"tps":10762.5,"type":"step"}],[31.626,{"elapsed":29.5671,"eta":16.2142,"grad_norm":0.277551,"loss":1.49138,"lr":0.000228396,"step":155,"total":240,"tps":10005.7,"type":"step"}],[32.701,{"elapsed":30.6419,"eta":15.321,"grad_norm":0.284619,"loss":1.48904,"lr":0.000211417,"step":160,"total":240,"tps":9527.39,"type":"step"}],[33.324,{"best":true,"bpb":0.359355,"step":160,"type":"eval","val_loss":1.48028}],[34.257,{"elapsed":31.5709,"eta":14.3504,"grad_norm":0.258855,"loss":1.47835,"lr":0.000195,"step":165,"total":240,"tps":11023.4,"type":"step"}],[35.26,{"elapsed":32.5735,"eta":13.4126,"grad_norm":0.282842,"loss":1.48151,"lr":0.000179224,"step":170,"total":240,"tps":10213.4,"type":"step"}],[36.313,{"elapsed":33.6308,"eta":12.4914,"grad_norm":0.270744,"loss":1.47738,"lr":0.000164163,"step":175,"total":240,"tps":9684.49,"type":"step"}],[37.344,{"elapsed":34.6608,"eta":11.5536,"grad_norm":0.26953,"loss":1.48034,"lr":0.000149888,"step":180,"total":240,"tps":9942.18,"type":"step"}],[37.41,{"step":180,"text":". Bunica citește vechea fântână, și construiește un pod de piatră în bibliotecă. Croitoreasa așteaptă luna plină, dar ascultă vechea fântână în bibliotecă. \nCălătorii ascultă o lampă albastră, apoi vede o poveste lungă la școală. Ana construiește vechea fântână luna plină, și caută un telescop mic dimineața. Inginerul ascultă o carte veche, și poartă o scrisoare. \nCercetătorii găsesc un telescop mic lângă fereastră. Mihai așteaptă o lampă albastră, și poartă un telescop mic dimineața. Pisica citește o lampă albastră. \nAna găsește o lampă albastră pe munte. Păsările repară un telescop mic. Mihai repară un pod de piatră, apoi desenează o carte veche în parc. \nVulpea așteaptă harta orașului, dar desenează o scrisoare în parc. Vulpea repară o scrisoare. Bunica vede un telescop mic în bibliotecă. Meșterii ascultă vechea fântână. \nAstronomul citește un","type":"sample"}],[38.381,{"elapsed":35.6288,"eta":10.5923,"grad_norm":0.282979,"loss":1.47113,"lr":0.000136469,"step":185,"total":240,"tps":10578.5,"type":"step"}],[39.368,{"elapsed":36.6189,"eta":9.63656,"grad_norm":0.277792,"loss":1.47115,"lr":0.000123967,"step":190,"total":240,"tps":10342.0,"type":"step"}],[40.282,{"elapsed":37.5331,"eta":8.66148,"grad_norm":0.266468,"loss":1.46729,"lr":0.000112443,"step":195,"total":240,"tps":11201.3,"type":"step"}],[41.149,{"elapsed":38.3997,"eta":7.67993,"grad_norm":0.288197,"loss":1.46677,"lr":0.000101952,"step":200,"total":240,"tps":11816.7,"type":"step"}],[41.735,{"best":true,"bpb":0.356677,"step":200,"type":"eval","val_loss":1.46925}],[42.729,{"elapsed":39.3936,"eta":6.72573,"grad_norm":0.266082,"loss":1.47571,"lr":9.25421e-05,"step":205,"total":240,"tps":10302.6,"type":"step"}],[43.718,{"elapsed":40.3823,"eta":5.76891,"grad_norm":0.284124,"loss":1.47123,"lr":8.42591e-05,"step":210,"total":240,"tps":10356.4,"type":"step"}],[44.658,{"elapsed":41.322,"eta":4.80488,"grad_norm":0.253772,"loss":1.46775,"lr":7.7142e-05,"step":215,"total":240,"tps":10898.0,"type":"step"}],[45.636,{"elapsed":42.3,"eta":3.84546,"grad_norm":0.266436,"loss":1.46513,"lr":7.12246e-05,"step":220,"total":240,"tps":10469.9,"type":"step"}],[46.641,{"elapsed":43.3018,"eta":2.88679,"grad_norm":0.26405,"loss":1.45982,"lr":6.6535e-05,"step":225,"total":240,"tps":10221.6,"type":"step"}],[47.716,{"elapsed":44.3802,"eta":1.92957,"grad_norm":0.250171,"loss":1.4682,"lr":6.30954e-05,"step":230,"total":240,"tps":9495.49,"type":"step"}],[48.677,{"elapsed":45.3415,"eta":0.964712,"grad_norm":0.262199,"loss":1.45976,"lr":6.09222e-05,"step":235,"total":240,"tps":10653.0,"type":"step"}],[49.588,{"elapsed":46.2519,"eta":0.0,"grad_norm":0.270103,"loss":1.45854,"lr":6e-05,"step":240,"total":240,"tps":11247.0,"type":"step"}],[50.293,{"best":true,"bpb":0.353853,"step":240,"type":"eval","val_loss":1.45761}],[50.325,{"step":240,"type":"saved"}],[50.392,{"step":240,"text":". Inginerul desenează vechea fântână în bibliotecă. Călătorii găsesc o lampă albastră. Vulpea așteaptă un copac înalt. Pescarul găsește harta orașului, și poartă o poveste lungă. Astronomul vede o scrisoare, apoi repară o scrisoare. Pisica vede harta orașului, și așteaptă luna plină în parc. \nAna găsește harta orașului, dar citește luna plină. \nAna desenează o lampă albastră, și vede vechea fântână. Profesorul caută o poveste lungă în parc. \nElevii citesc un pod de piatră, apoi ascultă o carte veche lângă fereastră. Mihai poartă o lampă albastră, și ascultă un copac înalt. Ana caută pădurea întunecată. \nDoctorița desenează o scrisoare. \nVulpea așteaptă pădurea întunecată, dar poartă un pod de piatră. Meșterii găsesc o scrisoare, apoi poartă un telescop mic dimineața. Păsările repară o poveste lungă în bibliotecă. \nMihai poartă o","type":"sample"}],[50.393,{"error":null,"result":{"best_val_loss":1.45761,"bytes_per_token":5.94285,"elapsed_secs":46.2519,"final_step":240,"interrupted":false,"last_train_loss":1.44275,"last_val_loss":1.45761,"name":"demo-nano","steps_run":240,"tokens_per_sec":10627.0,"tokens_seen":491520},"state":"done","type":"end"}]]},
  generate: {"loaded":{"context":256,"load_secs":0.0136322,"params":933760,"step":240,"tokenizer":"bpe","type":"loaded","vocab_size":571},"text":" un pod de piatră, apoi citește o scrisoare. \nAstronomul desenează un telescop mic, și desenează o lampă albastră. Pescarul construiește vechea fântână, și desenează o lampă albastră. Copilul găsește vechea fântână. Pescarul desenează luna plină, iar apoi citește vechea fântână pe munte. \nPisica vede un cântec frumos, dar găsește o scrisoare în bibliotecă. Astronomul desenează o carte veche. Copilul repară o lampă albastră în bibliotecă. Vulpea caută o scrisoare, apoi caută o lampă albastră. \nBunica poartă un cântec frumos în fața casei. Doctorița vede un telescop mic. Vecinii așteaptă un copac înalt, dar poartă harta orașului. \nPescarul așteaptă un pod de piatră în fața casei. Croitoreasa poartă o poveste lungă, și desenează un telescop mic în fața casei. Bunica citește o carte veche pe munte. Cercetătorii ascultă pădurea întunecată în parc. \nProfesorul desenează o lampă albastră. Pisica construiește un pod de piatră în parc. Bunica poartă o scrisoare. Bunica desenează pădurea întunecată. Mihai găsește un telescop mic, iar harta orașului. \nBunica construiește un telescop mic. Profesorul desenează un telescop mic, dar găsește un pod de piatră. Inginerul ascultă o poveste lungă pe malul râului. Pescarul citește un telescop mic pe munte. Doctorița caută un telescop mic, dar vede o carte veche la școală. \nMeșterii repară un telescop mic. Vulpea repară o lampă albastră acasă. Profesorul vede o lampă albastră pe munte. Vecinii așteaptă o scrisoare. Meșterii așteaptă o poveste lungă, repară un pod de piatră. \nPăsările desenează o lampă albastră, apoi citesc un telescop mic. Copilul găsește pădurea întunecată, apoi repară vechea fântână în bibliotecă. Pescarul construiește luna plină, și găsește un copac înalt. \nCopilul poartă un cântec frumos în fața casei. Bunica desenează vechea fântână în bibliotecă. Mihai construiește un cântec frumos în bibliotecă. \nBunica așteaptă luna plină, dar citește vechea fântână în bibliotecă. Mihai caută o lampă albastră, și vede o carte veche. Astronomul repară luna plină. \nPisica repară o scrisoare, dar așteaptă un cântec frumos. Astronomul citește o carte veche, iar apoi găsește o lampă albastră. Meșterii găsesc luna plină la școală. \nPescarul citește o scrisoare, dar construiește o scrisoare. Mihai așteaptă vechea fântână. Pescarul vede o carte veche, apoi poartă un copac înalt. Pescarul citește o lampă albastră în bibliotecă. \nInginerul construiește o lampă albastră pe munte. Astronomul caută o scrisoare. Pisica citește o poveste lungă în bibliotecă. Păsările repară o carte veche, dar caută o poveste lungă. \nAna construiește o scrisoare, iar apoi caută luna plină. Doctorița vede un telescop mic, și construiește un pod de piatră pe malul râului. Profesorul construiește un copac înalt, dar desenează un copac înalt. \nAna citește o carte veche. \nMihai vede o carte veche pe munte. Meșterii găsesc un copac înalt. Profesorul găsesc pădurea întunecată, apoi ascultă un pod de piatră. \nCopilul construiește pădurea întunecată, iar apoi construiește un pod de piatră. Bunica desenează o scrisoare la școală. Pisica caută vechea fântână pe munte. Ana ascultă o poveste lungă, dar ascultă o scrisoare. \nMihai poartă vechea fântână. Profesorul construiește o carte veche acasă. Doctorița repară vechea fântână. Pisica poartă o scrisoare acasă. Bunica desenează un pod de piatră. \nCercetătorii citește un pod de piatră. Doctorița caută luna plină. Croitoreasa construiește un cântec frumos la școală. Croitoreasa construiește o carte veche. Vulpea ascultă o lampă albastră acasă. \nPisica așteaptă harta orașului. Copilul poartă o scrisoare. Vulpea repară o scrisoare, iar apoi vede un pod de piatră. \nPescarul citește o lampă albastră, apoi vede o scrisoare la școală. Pescarul caută o scrisoare, apoi"},
  bench: {"results":[{"batch":8,"gflops":59.2713,"ms_per_step":98.5041,"ms_per_step_best":93.1686,"params":853120,"peak_rss_mib":261.309,"preset":"nano","replicas":2,"seq":128,"steps":2,"tokens_per_sec":10395.5,"secs":0.407},{"batch":8,"gflops":66.1017,"ms_per_step":278.682,"ms_per_step_best":269.653,"params":2755008,"peak_rss_mib":415.453,"preset":"micro","replicas":2,"seq":128,"steps":2,"tokens_per_sec":3674.43,"secs":0.98},{"batch":8,"gflops":71.6102,"ms_per_step":969.941,"ms_per_step_best":916.458,"params":10818432,"peak_rss_mib":700.66,"preset":"mini","replicas":2,"seq":128,"steps":2,"tokens_per_sec":1055.73,"secs":3.195},{"batch":8,"gflops":88.1693,"ms_per_step":1835.86,"ms_per_step_best":1730.52,"params":25436672,"peak_rss_mib":1164.69,"preset":"small","replicas":2,"seq":128,"steps":2,"tokens_per_sec":557.776,"secs":5.984}],"scaling":[{"efficiency":1.0,"result":{"batch":8,"gflops":35.7692,"ms_per_step":163.226,"ms_per_step_best":163.02,"params":853120,"peak_rss_mib":1164.69,"preset":"nano","replicas":1,"seq":128,"steps":2,"tokens_per_sec":6273.5},"speedup":1.0,"threads":1},{"efficiency":0.759484,"result":{"batch":8,"gflops":54.3322,"ms_per_step":107.459,"ms_per_step_best":94.905,"params":853120,"peak_rss_mib":1164.69,"preset":"nano","replicas":2,"seq":128,"steps":2,"tokens_per_sec":9529.24},"speedup":1.51897,"threads":2}]}
};
/* RECORDING:END */

const DEMO = {
  jobs: new Map(),
  next: 1,
  models: [],
  // what a first visit sees, with the training form already set to what was recorded
  settings: {
    bench: {},
    contrast: 'auto',
    density: 'comfortable',
    form: { t_preset: 'nano', t_tok: 'bpe', t_vocab: 2048, t_steps: 240, t_batch: 8, t_warm: 12, t_evalev: 40, t_log: 5, t_sample: 60, t_mb: 0.5, t_source: 'demo' },
    lang: 'auto',
    motion: 'auto',
    notify: false,
    rail: 'full',
    sound: false,
    text_scale: 1.0,
    theme: 'auto',
  },
};

// a few folders with a few texts, for the file picker
const DEMO_HOME = '/home/demo';
const DEMO_TREE = {
  [DEMO_HOME]: [['GERMEN', 0], ['Documents', 0], ['Desktop', 0]],
  [`${DEMO_HOME}/GERMEN`]: [['data', 0], ['models', 0]],
  [`${DEMO_HOME}/GERMEN/data`]: [['povesti-pentru-copii.txt', 301_244], ['jurnal-de-bord.txt', 184_960]],
  [`${DEMO_HOME}/Documents`]: [['carte.txt', 923_480], ['notite.txt', 21_310]],
  [`${DEMO_HOME}/Desktop`]: [],
};

const demoNow = () => Math.floor(Date.now() / 1000);
const demoFail = (status, code, message, params) => new ApiFailure(status, code, message, params);

function installDemo() {
  DEMO.models = [{ ...REC.model, modified: demoNow() - 2 * 86400 }];
  api.impl = demoRequest;
  api.upload = demoUpload;
}

// ---------------------------------------------------------------------------- requests

const DEMO_ROUTES = [
  ['GET', /^\/api\/ping$/, () => ({ ok: true })],
  ['GET', /^\/api\/system$/, () => REC.system],
  ['GET', /^\/api\/settings$/, () => DEMO.settings],
  ['PUT', /^\/api\/settings$/, (m, body) => demoSettings(body)],
  ['GET', /^\/api\/models$/, () => ({ models: DEMO.models })],
  ['GET', /^\/api\/jobs$/, () => ({ jobs: [...DEMO.jobs.values()].reverse().map(demoSummary) })],
  ['GET', /^\/api\/jobs\/(\d+)$/, (m) => demoSnapshot(demoJob(m[1]))],
  ['GET', /^\/api\/jobs\/(\d+)\/events$/, (m, body, q, o) => demoEvents(demoJob(m[1]), Number(q.get('after')) || 0, Number(q.get('wait')) || 0, o.signal)],
  ['POST', /^\/api\/jobs\/(\d+)\/stop$/, (m) => demoStop(demoJob(m[1]))],
  ['POST', /^\/api\/train$/, (m, body) => demoTrain(body)],
  ['POST', /^\/api\/generate$/, (m, body) => demoGenerate(body)],
  ['POST', /^\/api\/bench$/, (m, body) => demoBench(body)],
  ['GET', /^\/api\/estimate$/, (m, body, q) => demoEstimate(q)],
  ['GET', /^\/api\/fs$/, (m, body, q) => demoFolder(q.get('path') || '')],
  ['POST', /^\/api\/verify$/, (m, body) => demoVerify(body)],
  ['POST', /^\/api\/recover$/, () => ({ files: 0, kind: 'nothing' })],
  ['POST', /^\/api\/quit$/, () => ({ ok: true })],
];

async function demoRequest(method, path, body, options = {}) {
  if (options.signal?.aborted) throw new DOMException('aborted', 'AbortError');
  const url = new URL(path, 'http://demo.invalid');
  const found = DEMO_ROUTES.find(([verb, pattern]) => verb === method && pattern.test(url.pathname));
  if (!found) throw demoFail(404, 'not_found', 'Not found');
  if (!url.pathname.endsWith('/events')) await sleep(25 + Math.random() * 45); // a program takes a moment to answer
  const answer = await found[2](found[1].exec(url.pathname), body, url.searchParams, options);
  return JSON.parse(JSON.stringify(answer)); // what comes over the wire is a copy, never the page's own objects
}

function demoUpload(file, onProgress, signal) {
  return new Promise((resolve, reject) => {
    if (!file.size) return reject(demoFail(422, 'empty', 'That file is empty.'));
    let done = 0;
    const timer = setInterval(() => {
      done += 0.25;
      onProgress?.(Math.min(done, 1));
      if (done >= 1) {
        clearInterval(timer);
        resolve({ bytes: file.size, name: file.name, path: `${REC.system.data_dir}/${file.name}` });
      }
    }, 90);
    signal?.addEventListener('abort', () => {
      clearInterval(timer);
      reject(new DOMException('aborted', 'AbortError'));
    });
  });
}

/** Like the program, the page keeps its settings; unlike it, only until it is closed (no browser storage is used). */
function demoSettings(change) {
  Object.assign(DEMO.settings, change);
  return DEMO.settings;
}

// ---------------------------------------------------------------------------- jobs

function demoNew(kind, label, request) {
  const parts = {
    train: { stage: { name: 'queued' }, start: null, last: null, steps: { step: [], loss: [], lr: [], tps: [], grad_norm: [] }, evals: { step: [], val_loss: [], bpb: [], best: [] }, samples: [], log: [] },
    bench: { plan: null, current: null, results: [], points: [] },
    generate: { loaded: null, text: '' },
  }[kind];
  const job = { id: DEMO.next, kind, label, request, started: demoNow(), state: kind === 'train' ? 'starting' : 'running', events: [], waiters: new Set(), timers: [], clock: performance.now(), result: null, error: null, parts };
  DEMO.next += 1;
  DEMO.jobs.set(job.id, job);
  return job;
}

function demoJob(id) {
  const job = DEMO.jobs.get(Number(id));
  if (!job) throw demoFail(404, 'not_found', 'Not found');
  return job;
}

const demoClock = (job) => (performance.now() - job.clock) / 1000;

/** Runs `action` after `seconds`; the job keeps the timer so that a stop can cancel it. */
function demoLater(job, seconds, action) {
  job.timers.push(setTimeout(action, Math.max(0, seconds * 1000)));
}

function demoCancel(job) {
  for (const timer of job.timers) clearTimeout(timer);
  job.timers = [];
}

function demoEmit(job, e) {
  job.events.push({ seq: job.events.length, e });
  demoFold(job, e);
  for (const wake of [...job.waiters]) wake();
}

/** Keeps the snapshot of a job up to date with the events, as the real program does. */
function demoFold(job, e) {
  const p = job.parts;
  const at = demoClock(job);
  if (e.type === 'end') {
    job.state = e.state;
    job.result = e.result;
    job.error = e.error;
  }
  if (job.kind === 'train') {
    if (e.type === 'stage') {
      p.stage = { name: e.name, done: e.done, total: e.total };
      if (e.total === null) p.log.push({ t: at, type: 'stage', name: e.name });
    } else if (e.type === 'start') {
      p.start = e;
      p.stage = { name: 'training' };
      job.state = 'running';
      p.log.push({ t: at, type: 'start' });
    } else if (e.type === 'step') {
      p.last = e;
      for (const key of Object.keys(p.steps)) p.steps[key].push(e[key]);
    } else if (e.type === 'eval') {
      p.evals.step.push(e.step);
      p.evals.val_loss.push(e.val_loss);
      p.evals.bpb.push(e.bpb);
      p.evals.best.push(e.best ? 1 : 0);
    } else if (e.type === 'sample') {
      p.samples.push({ step: e.step, text: e.text });
    } else if (e.type === 'saved' || e.type === 'stopping') {
      p.log.push({ t: at, type: e.type, step: e.step });
    }
  } else if (job.kind === 'bench') {
    if (e.type === 'bench_plan') p.plan = e;
    else if (e.type === 'bench_running') p.current = { preset: e.preset };
    else if (e.type === 'bench_result') {
      p.results.push(e.result);
      p.current = null;
    } else if (e.type === 'scaling_point') {
      p.points.push(e);
      p.current = null;
    }
  } else if (e.type === 'loaded') {
    p.loaded = e;
  } else if (e.type === 'text') {
    p.text += e.text;
  }
}

function demoSummary(job) {
  const latest = job.parts.last;
  return { id: job.id, kind: job.kind, label: job.label, started: job.started, state: job.state, ...(latest ? { step: latest.step, total: latest.total } : {}) };
}

function demoSnapshot(job) {
  return { id: job.id, kind: job.kind, label: job.label, request: job.request, state: job.state, started: job.started, elapsed: demoClock(job), seq: job.events.length, result: job.result, error: job.error, [job.kind]: job.parts };
}

/** A long poll: answers at once if there is something after `after`, otherwise when there is, or after `wait` ms. */
async function demoEvents(job, after, wait, signal) {
  if (job.events.length <= after && !FINAL.has(job.state)) {
    await new Promise((resolve) => {
      let timer = 0;
      const done = () => {
        clearTimeout(timer);
        job.waiters.delete(done);
        signal?.removeEventListener('abort', done);
        resolve();
      };
      timer = setTimeout(done, wait);
      job.waiters.add(done);
      signal?.addEventListener('abort', done);
    });
  }
  if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
  return { events: job.events.slice(after), next: job.events.length, state: job.state, gap: false };
}

function demoBusy(kinds) {
  const busy = [...DEMO.jobs.values()].find((job) => kinds.includes(job.kind) && !FINAL.has(job.state));
  if (busy) throw demoFail(409, 'busy', 'busy', { kind: busy.kind });
}

function demoStop(job) {
  if (FINAL.has(job.state) || job.state === 'stopping') return { ok: true };
  demoCancel(job);
  if (job.kind === 'train') {
    job.state = 'stopping';
    demoEmit(job, { type: 'stopping' });
    demoLater(job, 0.7, () => {
      const step = job.parts.last?.step || 0;
      demoEmit(job, { type: 'saved', step });
      demoEmit(job, { type: 'end', state: 'stopped', error: null, result: demoTrainResult(job, true) });
      demoRemember(job);
    });
  } else if (job.kind === 'generate') {
    const tokens = Math.round(job.parts.text.length / 5.4);
    demoEmit(job, { type: 'end', state: 'stopped', error: null, result: { interrupted: true, secs: demoClock(job), seed: job.seed, tokens, tokens_per_sec: tokens / Math.max(0.05, demoClock(job)) } });
  } else {
    demoEmit(job, { type: 'end', state: 'stopped', error: null, result: null });
  }
  return { ok: true };
}

// ---------------------------------------------------------------------------- training

const DEMO_RESUME_FROM = 160; // a continued training replays the end of the recording, where the loss is already low

function demoTrain(request) {
  demoBusy(['train', 'bench']);
  const name = String(request.name || '');
  const existing = DEMO.models.find((m) => m.name === name);
  if (request.mode === 'new' && existing) throw demoFail(409, 'model_exists', 'exists', { name, step: existing.step });
  if (request.mode === 'resume' && !existing) throw demoFail(404, 'model_missing', 'missing', { name });
  const from = request.mode === 'resume' ? existing.step : 0;
  const job = demoNew('train', name, request);
  job.from = from;
  for (const [seconds, e] of demoTrainEvents(from)) demoLater(job, seconds, () => demoPlay(job, e));
  return { job: job.id };
}

/** The recording as the events of this run: all of it for a new model, its end for a continued one. */
function demoTrainEvents(from) {
  const all = REC.train.events;
  if (!from) return all;
  const shift = from - DEMO_RESUME_FROM;
  const cut = all.findLast(([, e]) => e.type === 'step' && e.step <= DEMO_RESUME_FROM)?.[0] || 0;
  const out = [];
  for (const [seconds, e] of all) {
    const keep = e.type === 'stage' || e.type === 'start' || (e.step !== undefined && e.step > DEMO_RESUME_FROM) || e.type === 'end';
    if (!keep) continue;
    const moved = { ...e };
    if (e.type === 'start') Object.assign(moved, { from_step: from, resume: true, optimizer_restored: true, total_steps: from + (e.total_steps - DEMO_RESUME_FROM) });
    else if (e.type === 'step') Object.assign(moved, { step: e.step + shift, total: e.total + shift, elapsed: e.elapsed - all.find(([, x]) => x.type === 'step' && x.step > DEMO_RESUME_FROM)[1].elapsed + 0.8 });
    else if (e.type === 'eval' || e.type === 'sample' || e.type === 'saved') moved.step = e.step + shift;
    else if (e.type === 'end') moved.result = { ...e.result, final_step: e.result.final_step + shift, steps_run: e.result.final_step - DEMO_RESUME_FROM };
    out.push([e.type === 'stage' || e.type === 'start' ? seconds : Math.max(0.3, seconds - cut), moved]);
  }
  return out;
}

function demoPlay(job, e) {
  if (FINAL.has(job.state) || job.state === 'stopping') return;
  demoEmit(job, e.type === 'end' ? { ...e, result: { ...e.result, name: job.label } } : e);
  if (e.type === 'end') demoRemember(job);
}

function demoTrainResult(job, interrupted) {
  const p = job.parts;
  const end = p.last || {};
  const lastEval = p.evals.val_loss.length ? p.evals.val_loss[p.evals.val_loss.length - 1] : null;
  const done = REC.train.events[REC.train.events.length - 1][1].result;
  return {
    ...done,
    name: job.label,
    interrupted,
    final_step: end.step || job.from,
    steps_run: (end.step || job.from) - job.from,
    elapsed_secs: end.elapsed || 0,
    last_train_loss: end.loss ?? null,
    last_val_loss: lastEval,
    best_val_loss: p.evals.val_loss.length ? Math.min(...p.evals.val_loss) : null,
    tokens_seen: (end.step || 0) * (p.start?.tokens_per_step || 2048),
  };
}

/** A finished (or stopped) training leaves a model in the list. */
function demoRemember(job) {
  const result = job.result;
  if (!result || !(result.final_step > 0)) return;
  const entry = {
    ...REC.model,
    name: job.label,
    path: `${REC.system.models_dir}/${job.label}`,
    step: result.final_step,
    modified: demoNow(),
    elapsed_secs: result.elapsed_secs,
    last_train_loss: result.last_train_loss,
    last_val_loss: result.last_val_loss,
    best_val_loss: result.best_val_loss,
    tokens_seen: result.tokens_seen,
  };
  const at = DEMO.models.findIndex((m) => m.name === job.label);
  if (at >= 0) DEMO.models[at] = entry;
  else DEMO.models.unshift(entry);
}

// ---------------------------------------------------------------------------- writing

function demoGenerate(request) {
  demoBusy(['generate']);
  const model = DEMO.models.find((m) => m.path === request.model || m.name === request.model);
  if (!model) throw demoFail(404, 'model_missing', 'missing', { name: String(request.model || '') });
  const job = demoNew('generate', model.name, request);
  job.seed = Number.isFinite(request.seed) ? request.seed : Math.floor(Math.random() * 2 ** 31);
  const text = REC.generate.text.slice(0, Math.min(REC.generate.text.length, Math.round((request.max_new || 200) * 5.4)));
  demoLater(job, 0.25, () => demoEmit(job, { ...REC.generate.loaded, step: model.step, params: model.params }));
  const piece = 14;
  const every = 0.045;
  for (let i = 0; i * piece < text.length; i += 1) {
    demoLater(job, 0.4 + i * every, () => demoEmit(job, { type: 'text', text: text.slice(i * piece, (i + 1) * piece) }));
  }
  const total = 0.4 + Math.ceil(text.length / piece) * every;
  demoLater(job, total, () => {
    const tokens = request.max_new || 200;
    demoEmit(job, { type: 'end', state: 'done', error: null, result: { interrupted: false, secs: total, seed: job.seed, tokens, tokens_per_sec: tokens / total } });
  });
  return { job: job.id };
}

// ---------------------------------------------------------------------------- speed

/** The recorded numbers for a size; the one size that was not recorded is worked out from the one next to it. */
function demoSizeResult(name, request) {
  const recorded = REC.bench.results.find((r) => r.preset === name);
  const preset = REC.system.presets.find((p) => p.name === name);
  const reference = recorded || REC.bench.results[REC.bench.results.length - 1];
  const tps = recorded ? recorded.tokens_per_sec : (reference.tokens_per_sec * reference.params) / preset.params;
  const ms = ((request.batch * request.seq) / tps) * 1000;
  return {
    batch: request.batch,
    gflops: reference.gflops,
    ms_per_step: ms,
    ms_per_step_best: ms * 0.94,
    params: recorded ? recorded.params : preset.params,
    peak_rss_mib: recorded ? recorded.peak_rss_mib : preset.train_bytes / 1048576,
    preset: name,
    replicas: REC.system.workers,
    seq: request.seq,
    steps: request.steps,
    tokens_per_sec: tps,
  };
}

/** How much faster `threads` threads are than one: the recording measured one and two. */
function demoSpeedup(threads) {
  const two = REC.bench.scaling.find((p) => p.threads === 2)?.speedup || 1.7;
  if (threads <= 1) return 1;
  if (threads === 2) return two;
  return two * Math.max(0.6, 1 - 0.03 * Math.log2(threads / 2)); // no more cores than the recorded computer has
}

function demoBench(request) {
  demoBusy(['train', 'bench']);
  const names = request.preset === 'all' ? REC.bench.results.map((r) => r.preset) : [request.preset];
  const scaling = Array.isArray(request.scaling) ? request.scaling : null;
  const job = demoNew('bench', request.preset, request);
  demoEmit(job, { type: 'bench_plan', batch: request.batch, seq: request.seq, steps: request.steps, warmup: request.warmup, scaling: !!scaling, presets: names, optimised_build: true });
  let at = 0.05;
  const results = [];
  const points = [];
  for (const name of names) {
    const size = demoSizeResult(name, request);
    const secs = Math.min(9, REC.bench.results.find((r) => r.preset === name)?.secs || 6);
    demoLater(job, at, () => demoEmit(job, { type: 'bench_running', preset: name }));
    if (scaling) {
      const single = size.tokens_per_sec / demoSpeedup(REC.system.workers);
      const each = Math.max(0.5, secs / Math.max(1, scaling.length));
      scaling.forEach((threads, i) => {
        const speedup = demoSpeedup(threads);
        const result = { ...size, replicas: threads, tokens_per_sec: single * speedup, ms_per_step: ((request.batch * request.seq) / (single * speedup)) * 1000 };
        const point = { preset: name, type: 'scaling_point', point: { efficiency: speedup / threads, result, speedup, threads } };
        points.push(point.point);
        demoLater(job, at + each * (i + 1), () => demoEmit(job, point));
      });
      at += each * scaling.length + 0.05;
    } else {
      results.push(size);
      at += secs;
      demoLater(job, at, () => demoEmit(job, { type: 'bench_result', result: size }));
    }
  }
  demoLater(job, at + 0.05, () => {
    const report = { germen_version: REC.system.version, logical_cpus: REC.system.logical_cpus, optimised_build: true, platform: REC.system.platform, results, scaling: scaling ? names.map((preset) => ({ preset, points })) : [] };
    demoEmit(job, { type: 'end', state: 'done', error: null, result: { report } });
  });
  return { job: job.id };
}

// ---------------------------------------------------------------------------- the rest

function demoEstimate(q) {
  const preset = REC.system.presets.find((p) => p.name === q.get('preset'));
  if (!preset) throw demoFail(400, 'invalid_field', 'preset', { field: 'preset' });
  const vocab = Number(q.get('vocab')) || 256;
  const tokens = (Number(q.get('batch')) || 16) * (Number(q.get('seq')) || 256);
  const params = preset.params + (vocab - 256) * preset.d_model * 2;
  const train = params * 16 + tokens * preset.d_model * preset.layers * 340;
  const fits = train < REC.system.memory.limit;
  return {
    corpus_fits: true,
    corpus_problem: null,
    d_model: preset.d_model,
    fits,
    flops_per_token: 6 * params,
    heads: preset.heads,
    layers: preset.layers,
    memory_limit_bytes: REC.system.memory.limit,
    params,
    problem: null,
    replicas: Number(q.get('replicas')) || REC.system.workers,
    train_bytes: train,
    weights_bytes: params * 4,
  };
}

function demoFolder(path) {
  const here = path.replace(/\/+$/, '') || REC.system.data_dir;
  const roots = [
    { kind: 'data', path: REC.system.data_dir },
    { kind: 'models', path: REC.system.models_dir },
    { kind: 'home', path: DEMO_HOME },
    { kind: 'drive', path: '/' },
  ];
  let entries;
  if (here === REC.system.models_dir) entries = DEMO.models.map((m) => ({ dir: true, model: true, modified: m.modified, name: m.name, size: 0 }));
  else if (DEMO_TREE[here]) entries = DEMO_TREE[here].map(([name, size]) => ({ dir: !size, model: false, modified: demoNow() - 86400 * 9, name, size }));
  else if (here === '/') entries = [{ dir: true, model: false, modified: demoNow(), name: 'home', size: 0 }];
  else if (here === '/home') entries = [{ dir: true, model: false, modified: demoNow(), name: 'demo', size: 0 }];
  else throw demoFail(404, 'not_a_folder', 'Not a folder');
  const parent = here === '/' ? null : here.replace(/\/[^/]*$/, '') || '/';
  return { entries, parent, path: here, roots, truncated: false };
}

function demoVerify(request) {
  const model = DEMO.models.find((m) => m.path === request.model || m.name === request.model);
  if (!model) throw demoFail(404, 'model_missing', 'missing', { name: String(request.model || '') });
  const files = [
    { bytes: 156, name: 'config.json' },
    { bytes: model.params * 8, name: 'optimizer.bpk' },
    { bytes: 3101, name: 'tokenizer.json' },
    { bytes: model.params * 4, name: 'weights.bpk' },
  ];
  return { files, leftovers: [], optimizer: true, params: model.params, step: model.step };
}
