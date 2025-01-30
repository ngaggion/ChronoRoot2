""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os 

def saveGraph(grafo, conf, image_name):
    path = os.path.join(conf['folders']['graphs'], image_name.replace('.png','.xml.gz'))

    if grafo is not False:
        g, pos, weight, clase, nodetype, age = grafo
        g.vertex_properties["pos"] = pos
        g.vertex_properties["nodetype"] = nodetype
        g.vertex_properties["age"] = age
        g.edge_properties["weight"] = weight
        g.edge_properties["clase"] = clase
        g.save(path)
    else:
        print('Not valid graph')
    return

def saveProps(image_name, it, grafo, csv_writer, number_lateral_roots):
    if grafo is not False:
        g, pos, weight, clase, nodetype, age = grafo
        
        main_root_len = 0
        sec_root_len = 0
        tot_len = 0
                    
        for i in g.edges():
            tot_len += weight[i]
            if clase[i][1] == 10:
                main_root_len += weight[i]
            else:
                sec_root_len += weight[i]
        
        row = [image_name, it, main_root_len, sec_root_len, number_lateral_roots, tot_len]
    else:
        row = [image_name, it, 0, 0, 0, 0]
        
    csv_writer.writerow(row)
    return
